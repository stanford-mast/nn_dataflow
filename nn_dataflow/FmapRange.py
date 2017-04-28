""" $lic$
Copyright (C) 2016-2017 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

If you use this program in your research, we request that you reference the
TETRIS paper ("TETRIS: Scalable and Efficient Neural Network Acceleration with
3D Memory", in ASPLOS'17. April, 2017), and that you send us a citation of your
work.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

from collections import namedtuple, Counter
import itertools
import numpy as np

_FMAP_POSITION_ATTRS = ['b', 'n', 'h', 'w']

'''
A position in a batched fmap.
'''
FmapPosition = namedtuple('FmapPosition', _FMAP_POSITION_ATTRS)


class FmapRange(object):
    '''
    A range of a batched fmap.
    '''

    def __init__(self, fp_beg, fp_end):
        for b, e in zip(fp_beg, fp_end):
            if b > e:
                raise ValueError('FmapRange: begin value > end value? '
                                 'beg: {}, end: {}'.format(fp_beg, fp_end))
        self.fp_beg = FmapPosition(*fp_beg)
        self.fp_end = FmapPosition(*fp_end)

    def _extract_attrs(self, *attrs):
        '''
        Extract the begin values and end values for the given attributes. Not
        specifying means all attributes.
        '''
        if not attrs:
            begs = self.fp_beg
            ends = self.fp_end
        else:
            begs = [getattr(self.fp_beg, a) for a in attrs]
            ends = [getattr(self.fp_end, a) for a in attrs]
        return begs, ends

    def beg_end(self, *attrs):
        '''
        Get the begin and end values for each of the given attributes. Return
        in the form of a list with (beg, end) for each attribute. Not
        specifying means all attributes.
        '''
        begs, ends = self._extract_attrs(*attrs)
        return zip(begs, ends)

    def range(self, *attrs):
        '''
        Generator for the range of the given attributes. Not specifying means
        all attributes.
        '''
        begs, ends = self._extract_attrs(*attrs)

        ranges = [range(b, e) for b, e in zip(begs, ends)]

        for tuple_ in itertools.product(*ranges):
            yield tuple_

    def size(self, *attrs):
        '''
        Get the total size of the fmap along the given attributes. Not
        specifying means all attributes.
        '''
        begs, ends = self._extract_attrs(*attrs)

        lens = [e - b for b, e in zip(begs, ends)]

        return np.prod(lens)

    def overlap(self, other):
        '''
        Get the overlap FmapRange of the two.
        '''
        if not isinstance(other, FmapRange):
            raise TypeError('FmapRange: an FmapRange object is required.')

        begs = []
        ends = []
        for srng, orng in zip(zip(self.fp_beg, self.fp_end),
                              zip(other.fp_beg, other.fp_end)):
            b = max(srng[0], orng[0])
            e = min(srng[1], orng[1])
            if b >= e:
                # No overlap, return 0 FmapRange.
                return FmapRange([0]*len(_FMAP_POSITION_ATTRS),
                                 [0]*len(_FMAP_POSITION_ATTRS))
            begs.append(b)
            ends.append(e)
        return FmapRange(begs, ends)

    def corresponding_input_fmap_range(self, layer):
        '''
        Get the corresponding input FmapRange.
        '''
        b_rng, h_rng, w_rng = self.beg_end('b', 'h', 'w')

        # Batch. Same as ofmap.
        b_src_beg, b_src_end = b_rng
        # Channel. All.
        n_src_beg = 0
        n_src_end = layer.nifm
        # Height tiling.
        # xy_i = xy_o * stride + (0 ... sfil-1)
        h_src_beg = h_rng[0] * layer.strd
        h_src_end = (h_rng[1] - 1) * layer.strd + layer.sfil
        assert h_src_end <= layer.hifm
        # Width tiling.
        w_src_beg = w_rng[0] * layer.strd
        w_src_end = (w_rng[1] - 1) * layer.strd + layer.sfil
        assert w_src_end <= layer.wifm

        return FmapRange(FmapPosition(b=b_src_beg, n=n_src_beg,
                                      h=h_src_beg, w=w_src_beg),
                         FmapPosition(b=b_src_end, n=n_src_end,
                                      h=h_src_end, w=w_src_end))

    def __contains__(self, fpos):
        '''
        Whether the given FmapPosition is in the FmapRange.
        '''
        return all(p >= b and p < e for p, b, e
                   in zip(fpos, self.fp_beg, self.fp_end))

    def __cmp__(self, other):
        if not isinstance(other, FmapRange):
            raise TypeError('FmapRange: an FmapRange object is required.')

        for srng, orng in zip(zip(self.fp_beg, self.fp_end),
                              zip(other.fp_beg, other.fp_end)):
            if srng[0] >= orng[1]:
                return 1
            elif srng[1] <= orng[0]:
                return -1
            elif srng != orng:
                raise ValueError('FmapRange: comparing two overlap ranges. '
                                 '{} vs. {}'.format(self, other))

        return 0

    def __str__(self):
        return 'beg: {}, end: {}'.format(self.fp_beg, self.fp_end)


class FmapRangeMap(object):
    '''
    A map with key as type FmapPosition, and the keys within a FmapRange all
    map to the same value.
    '''

    def __init__(self):
        self.keyvals = []

    def add(self, frng, val):
        '''
        Add an FmapRange, in which all the FmapPositions are mapped to the same
        value.

        All added FmapRanges must not overlap.
        '''
        if frng.size() == 0:
            return

        idx = sum([1 if kv[0] < frng else 0 for kv in self.keyvals])
        if idx < len(self.keyvals) and not (self.keyvals[idx][0] > frng):
            prev = self.keyvals[idx-1][0] if idx > 0 else None
            raise ValueError('FmapRangeMap: added FmapRange overlaps with '
                             'its next range. New: {}, prev: {}, next: {}.'
                             .format(str(frng), str(prev),
                                     str(self.keyvals[idx][0])))
        self.keyvals.insert(idx, (frng, val))
        # Ensure sorted.
        assert all(self.keyvals[idx][0] < self.keyvals[idx+1][0]
                   for idx in range(len(self.keyvals) - 1))

    def get(self, fpos):
        '''
        Get the value corresponding to the given FmapPosition.
        '''
        for kv in self.keyvals:
            if fpos in kv[0]:
                return kv[1]
        raise KeyError('FmapRangeMap: key {} is not found.'.format(fpos))

    def rget_counter(self, frng):
        '''
        Get the counts of values corresponding to each FmapPosition in the
        given FmapRange. Return a collections.Counter object.
        '''
        counts = Counter()
        for kv in self.keyvals:
            counts[kv[1]] = frng.overlap(kv[0]).size()
        return counts

    def rget_single(self, frng):
        '''
        Get the single value corresponding to the given FmapRange. The given
        FmapRange must only correspond to a single value. Otherwise raise a
        ValueError.
        '''
        counts = self.rget_counter(frng)
        for key in counts.keys():
            if counts[key] == frng.size():
                return key
        raise ValueError('FmapRange: given fmap range does not correspond to '
                         'a single value.')

