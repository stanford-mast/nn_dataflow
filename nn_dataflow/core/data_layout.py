""" $lic$
Copyright (C) 2016-2020 by Tsinghua University and The Board of Trustees of
Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

from collections import namedtuple
import itertools

from .fmap_range import FmapPosition, FmapRange, FmapRangeMap
from .node_region import NodeRegion
from .partition_scheme import PartitionScheme

DATA_LAYOUT_LIST = ['frngs',
                    'regions',
                    'parts',
                   ]

class DataLayout(namedtuple('DataLayout', DATA_LAYOUT_LIST)):
    '''
    The data layout for batched i/ofmap.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(DataLayout, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.frngs, tuple):
            raise TypeError('DataLayout: frngs must be a tuple.')
        for fr in ntp.frngs:
            if not isinstance(fr, FmapRange):
                raise TypeError('DataLayout: elements in frngs must be a '
                                'FmapRange object.')
        if not isinstance(ntp.regions, tuple):
            raise TypeError('DataLayout: regions must be a tuple.')
        for nr in ntp.regions:
            if not isinstance(nr, NodeRegion):
                raise TypeError('DataLayout: elements in regions must be a '
                                'NodeRegion object.')
        if not isinstance(ntp.parts, tuple):
            raise TypeError('DataLayout: parts must be a tuple.')
        for p in ntp.parts:
            if not isinstance(p, PartitionScheme):
                raise TypeError('DataLayout: elements in parts must be a '
                                'PartitionScheme object.')

        cls._validate_frngs(ntp.frngs)
        cls._validate_parts(ntp.parts, ntp.regions)

        if not len(ntp.frngs) == len(ntp.regions) == len(ntp.parts):
            raise ValueError('DataLayout: {} must have the same length.'
                             .format(', '.join(DATA_LAYOUT_LIST)))

        return ntp

    def complete_fmap_range(self):
        '''
        Get the complete FmapRange, i.e., a perfect hyper cube starting from
        origin point (0, ..., 0) with no holes.
        '''
        return FmapRange(self.frngs[0].fp_beg, self.frngs[-1].fp_end)

    def fmap_range_map(self):
        '''
        Get an `FmapRangeMap` instance, mapping from fmap range to absolute
        node coordinate.
        '''
        frmap = FmapRangeMap()

        for frng, region, part in zip(self.frngs, self.regions, self.parts):

            for pidx in part.gen_pidx():
                pcoord = part.coordinate(region, pidx)
                pfrng = part.fmap_range(frng, pidx)

                frmap.add(pfrng, pcoord)

        return frmap

    def nhops_to(self, fmap_range, *dest_list, **kwargs):
        '''
        Get the total number of hops to transfer the FmapRange `fmap_range` to
        destinations `dest_list` given as a list of absolute coordinates.

        If `forwarding` is True, the data can be forwarded between destinations
        rather than all from the source.
        '''
        forwarding = kwargs.pop('forwarding', False)
        if kwargs:
            raise ValueError('DataLayout: method nhops_to() got an unexpected '
                             'keyword argument: {}.'
                             .format(kwargs.popitem()[0]))

        # The number of hops to transfer data to each destination individually.
        nhops_list = [0] * len(dest_list)

        for frng, region, part in zip(self.frngs, self.regions, self.parts):

            # Skip non-overlapped fmap range.
            if fmap_range.overlap_size(frng) == 0:
                continue

            for pidx in part.gen_pidx():
                psrc = part.coordinate(region, pidx)
                pfrng = part.fmap_range(frng, pidx)
                size = fmap_range.overlap_size(pfrng)

                nhops_list = [n + size * d.hop_dist(psrc)
                              for n, d in zip(nhops_list, dest_list)]

        if forwarding:
            # The number of hops to the first node and its coordinate.
            nhops, coord = min(zip(nhops_list, dest_list))

            # Size of all data.
            total_size = self.complete_fmap_range().overlap_size(fmap_range)

            # Data can be forwarded from all sources to any destination.
            src_set = {coord}
            dst_set = set(dest_list) - src_set

            while dst_set:
                # Each forward step, get the min-distance pair of source and
                # destination.
                src, dst = min(itertools.product(src_set, dst_set),
                               key=lambda sd: sd[1].hop_dist(sd[0]))
                dst_set.remove(dst)
                src_set.add(dst)
                nhops += total_size * dst.hop_dist(src)

        else:
            nhops = sum(nhops_list)

        return nhops

    def is_in(self, *regions):
        '''
        Whether the layout is completely in the given NodeRegion's `regions`.
        Region types must match. Each fmap range can be split into multiple
        given regions.
        '''
        return all(any(region.type == r.type and r.contains_node(coord)
                       for r in regions)
                   for region in self.regions for coord in region.iter_node())

    @classmethod
    def concat(cls, *data_layout_list):
        '''
        Concatenate multiple `DataLayout` objects along the channel dimension.
        '''
        frngs = []
        regions = []
        parts = []

        n_offset = 0

        for dl in data_layout_list:

            # Check type.
            if not isinstance(dl, DataLayout):
                raise TypeError('DataLayout: only DataLayout object can be '
                                'concatenated.')

            # Concatenate frngs along n dimension.
            for frng in dl.frngs:
                fpb = frng.fp_beg
                fpe = frng.fp_end
                frng2 = FmapRange(FmapPosition(b=fpb.b, n=fpb.n + n_offset,
                                               h=fpb.h, w=fpb.w),
                                  FmapPosition(b=fpe.b, n=fpe.n + n_offset,
                                               h=fpe.h, w=fpe.w))
                frngs.append(frng2)
                n_offset += frng.size('n')

            # Regions and partitions are the same.
            regions += dl.regions
            parts += dl.parts

        return DataLayout(frngs=tuple(frngs), regions=tuple(regions),
                          parts=tuple(parts))

    @classmethod
    def _validate_frngs(cls, frngs):
        '''
        Validate the fmap ranges.
        '''
        if not frngs:
            raise ValueError('DataLayout: no frngs.')

        _, n_end = frngs[0].beg_end('n')
        bhw_beg_end = frngs[0].beg_end('b', 'h', 'w')

        if frngs[0].fp_beg != FmapPosition(0, 0, 0, 0):
            raise ValueError('DataLayout: frngs must begin at 0.')

        for frng in frngs[1:]:
            if frng.beg_end('b', 'h', 'w') != bhw_beg_end:
                raise ValueError('DataLayout: frng dim b, h, w mismatch.')
            nb, ne = frng.beg_end('n')
            if nb != n_end:
                raise ValueError('DataLayout: frng dim n is discontinuous.')
            n_end = ne

    @classmethod
    def _validate_parts(cls, parts, regions):
        '''
        Validate the partitioning schemes.
        '''
        for region, part in zip(regions, parts):
            if not part.is_applicable_to_fmap_range():
                raise ValueError('DataLayout: invalid partitioning scheme for '
                                 'fmap range.')

            if any(pd > rd for pd, rd in zip(part.dim(), region.dim)):
                raise ValueError('DataLayout: partitioning scheme does not fit '
                                 'in node region.')

