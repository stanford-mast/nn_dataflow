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

from collections import namedtuple

from .fmap_range import FmapPosition, FmapRange, FmapRangeMap
from .node_region import NodeRegion
from .phy_dim2 import PhyDim2

DATA_LAYOUT_LIST = ['frmap',
                    'origin',
                    'type',
                   ]

class DataLayout(namedtuple('DataLayout', DATA_LAYOUT_LIST)):
    '''
    The data layout for batched i/ofmap.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(DataLayout, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.frmap, FmapRangeMap):
            raise TypeError('DataLayout: frmap must be a FmapRangeMap object.')
        if not isinstance(ntp.origin, PhyDim2):
            raise TypeError('DataLayout: origin must be a PhyDim2 object.')
        if ntp.type not in range(NodeRegion.NUM):
            raise ValueError('DataLayout: type must be a valid NodeRegion '
                             'type enum.')

        if not ntp.frmap.is_complete():
            raise ValueError('DataLayout: frmap must be a complete map.')

        coords_list = [val for _, val in ntp.frmap.items()]
        for coords in coords_list:
            if not isinstance(coords, tuple) \
                    or not all(isinstance(c, PhyDim2) for c in coords):
                raise TypeError('DataLayout: frmap value must be a tuple of '
                                'PhyDim2 objects.')

        return ntp

    def total_transfer_nhops(self, frng, dst_coord):
        '''
        Get the total number of hops to transfer the FmapRange `frng` to
        destination coordinate `dst_coord`.
        '''
        dst_coord_ofs = dst_coord - self.origin

        coords_cnts = self.frmap.rget_counter(frng)
        # This assertion is invalid now, because layout is not padded while
        # frng is padded.
        # assert sum(coords_cnts.values()) == frng.size()

        nhops = 0
        for coords, cnt in coords_cnts.items():
            nhops += cnt * sum(dst_coord_ofs.hop_dist(c) for c in coords)
        return nhops

    def is_in_region(self, node_region):
        '''
        Whether the layout is in the given NodeRegion.
        '''
        if self.type != node_region.type:
            return False
        all_coords = []
        for coords in self.frmap.rget_counter(self.frmap.complete_fmap_range()):
            all_coords += coords
        return all(node_region.contains_node(c + self.origin)
                   for c in all_coords)

    def view(self, origin_diff=PhyDim2(0, 0)):
        '''
        Another view of the layout by shifting the origin by the given
        `origin_diff`.
        '''
        return DataLayout(frmap=self.frmap, origin=self.origin + origin_diff,
                          type=self.type)

    def merge(self, merge_symbol, other):
        '''
        Merge self with `other` DataLayout, according to the given
        `merge_symbol`. Return the merged layout.

        Currently support merge symbols:
        - |: concatenate fmaps along the channel (n) dimension.
        - +: sum up fmaps along the channel (n) dimension.
        '''

        if not isinstance(other, DataLayout):
            raise TypeError('DataLayout: other must be a DataLayout object.')

        if self.type != other.type:
            raise ValueError('DataLayout: layouts to be merged must have '
                             'the same type.')

        scfrng = self.frmap.complete_fmap_range()
        ocfrng = other.frmap.complete_fmap_range()

        coord_ofs = other.origin - self.origin

        frmap = None

        if merge_symbol == '|':
            # Concatenate fmaps.

            # Check dimension match.
            if scfrng.beg_end('b', 'h', 'w') != ocfrng.beg_end('b', 'h', 'w'):
                raise ValueError('DataLayout: |-merging layouts do not match.')

            frmap = self.frmap.copy()

            # Offset for n.
            nofs = scfrng.fp_end.n

            for ofrng, ocoords in other.frmap.items():
                fpb = ofrng.fp_beg
                fpe = ofrng.fp_end
                o2frng = FmapRange(FmapPosition(b=fpb.b, n=fpb.n + nofs,
                                                h=fpb.h, w=fpb.w),
                                   FmapPosition(b=fpe.b, n=fpe.n + nofs,
                                                h=fpe.h, w=fpe.w))
                o2coords = tuple(c + coord_ofs for c in ocoords)

                frmap.add(o2frng, o2coords)

        elif merge_symbol == '+':
            # Sum fmaps.

            # Check dimension match.
            if scfrng.beg_end() != ocfrng.beg_end():
                raise ValueError('DataLayout: +-merging layouts do not match.')

            frmap = FmapRangeMap()

            for ofrng, ocoords in other.frmap.items():
                o2coords = tuple(c + coord_ofs for c in ocoords)

                for sfrng, scoords in self.frmap.items():
                    frng = sfrng.overlap(ofrng)
                    if frng.size() > 0:
                        # Fetch data from both coordinates.
                        frmap.add(frng, scoords + o2coords)

        else:
            raise ValueError('DataLayout: unrecognized merge symbol {}'
                             .format(merge_symbol))

        assert frmap.is_complete()

        return DataLayout(frmap=frmap, origin=self.origin, type=self.type)

