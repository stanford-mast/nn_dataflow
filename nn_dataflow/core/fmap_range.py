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

from .. import util

_FMAP_POSITION_ATTRS = ['b', 'n', 'h', 'w']

'''
A position in a batched fmap.
'''
FmapPosition = namedtuple('FmapPosition', _FMAP_POSITION_ATTRS)


class FmapRange(util.ContentHashClass):
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

        return util.prod(lens)

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

    def __contains__(self, fpos):
        '''
        Whether the given FmapPosition is in the FmapRange.
        '''
        return all(p >= b and p < e for p, b, e
                   in zip(fpos, self.fp_beg, self.fp_end))

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self._compare(other) < 0
        return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self._compare(other) > 0
        return NotImplemented

    def __ge__(self, other):
        return self > other or self == other

    def _compare(self, other):
        # Identical or empty ranges.
        if (self.fp_beg == other.fp_beg and self.fp_end == other.fp_end) \
                or (self.size() == 0 and other.size() == 0):
            return 0

        # Overlap check.
        if self.overlap(other).size() > 0:
            raise ValueError('FmapRange: comparing two overlap ranges. '
                             '{} vs. {}'.format(self, other))

        # We compare the two range using their begin points.
        if self.fp_beg > other.fp_beg:
            return 1
        assert self.fp_beg < other.fp_beg
        return -1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'fp_beg={}'.format(repr(self.fp_beg)),
                'fp_end={}'.format(repr(self.fp_end))]))


class FmapRangeMap(object):
    '''
    A map with key as type FmapPosition, and the keys within a FmapRange all
    map to the same value.
    '''

    def __init__(self):
        self.keyvals = []
        self.min_fpos = [float('inf')] * len(_FMAP_POSITION_ATTRS)
        self.max_fpos = [-float('inf')] * len(_FMAP_POSITION_ATTRS)

    def add(self, frng, val):
        '''
        Add an FmapRange, in which all the FmapPositions are mapped to the same
        value.

        All added FmapRanges must not overlap.
        '''
        if frng.size() == 0:
            return

        try:
            idx = sum([1 if kv[0] < frng else 0 for kv in self.keyvals])
            assert not (idx < len(self.keyvals) \
                    and not (self.keyvals[idx][0] > frng))
        except ValueError:
            raise ValueError('FmapRangeMap: added FmapRange overlaps with '
                             'existing ranges. Added: {}, existing: {}.'
                             .format(str(frng),
                                     [k for k, _ in self.keyvals]))
        self.keyvals.insert(idx, (frng, val))

        self.min_fpos = [min(a, b) for a, b in zip(self.min_fpos, frng.fp_beg)]
        self.max_fpos = [max(a, b) for a, b in zip(self.max_fpos, frng.fp_end)]

        # Ensure sorted.
        assert all(self.keyvals[idx][0] < self.keyvals[idx+1][0]
                   for idx in range(len(self.keyvals) - 1))

    def complete_fmap_range(self):
        '''
        Get the complete FmapRange. If the map is not complete, raise a
        ValueError. Complete map means the map covers a perfect hyper cube
        starting from origin point (0, ..., 0) with no holes.
        '''
        cfrng = FmapRange(fp_beg=self.min_fpos, fp_end=self.max_fpos)
        if cfrng.fp_beg != FmapPosition(0, 0, 0, 0) \
                or cfrng.size() != sum(self.rget_counter(cfrng).values()):
            raise ValueError('FmapRangeMap: not a complete map.')
        return cfrng

    def is_complete(self):
        '''
        Whether the map is a complete map.
        '''
        try:
            self.complete_fmap_range()
        except ValueError:
            return False
        return True

    def get(self, fpos):
        '''
        Get the value corresponding to the given FmapPosition.
        '''
        for kv in self.keyvals:
            if fpos in kv[0]:
                return kv[1]
        raise KeyError('FmapRangeMap: key {} is not found.'.format(fpos))

    def items(self):
        '''
        A generator to iterate over all FmapRanges and their values. Yield the
        FmapRange and its value as a pair at a time.
        '''
        for kv in self.keyvals:
            yield kv[0], kv[1]

    def copy(self):
        '''
        Get a copy of the map.
        '''
        new = FmapRangeMap()
        for frng, val in self.items():
            new.add(frng, val)
        return new

    def rget_counter(self, frng):
        '''
        Get the counts of values corresponding to each FmapPosition in the
        given FmapRange. Return a collections.Counter object.
        '''
        counts = Counter()
        for kv in self.keyvals:
            counts[kv[1]] = counts.setdefault(kv[1], 0) \
                    + frng.overlap(kv[0]).size()
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
        raise ValueError('FmapRangeMap: given fmap range does not correspond '
                         'to a single value.')

    def __str__(self):
        str_ = '{}:\n'.format(self.__class__.__name__)
        for k, v in self.items():
            str_ += '  {} -> {}\n'.format(str(k), str(v))
        return str_

