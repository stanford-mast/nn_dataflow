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

import itertools
from collections import namedtuple

from .. import util
from .phy_dim2 import PhyDim2

NODE_REGION_LIST = ['dim',
                    'origin',
                    'type',
                   ]

class NodeRegion(namedtuple('NodeRegion', NODE_REGION_LIST)):
    '''
    A node region defined by the dimension and origin offset.

    The `type` attribute specifies the region type, which could be `PROC` for
    computation processing nodes or 'DATA' for data storage nodes.

    NOTE: we cannot overload __contains__ and __iter__ as a node container,
    because the base namedtuple already defines them.
    '''

    # Type enums.
    PROC = 0
    DATA = 1
    NUM = 2

    def __new__(cls, *args, **kwargs):
        ntp = super(NodeRegion, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.dim, PhyDim2):
            raise TypeError('NodeRegion: dim must be a PhyDim2 object.')
        if not isinstance(ntp.origin, PhyDim2):
            raise TypeError('NodeRegion: origin must be a PhyDim2 object.')

        if ntp.type not in range(cls.NUM):
            raise ValueError('NodeRegion: type must be a valid type enum.')

        return ntp

    def contains_node(self, coordinate):
        ''' Whether the region contains the given coordinate. '''
        min_coord = self.origin
        max_coord = self.origin + self.dim
        return all(cmin <= c and c < cmax for c, cmin, cmax
                   in zip(coordinate, min_coord, max_coord))

    def node_iter(self):
        ''' Iterate through all nodes in the region. '''
        gens = []
        for o, d in zip(self.origin, self.dim):
            gens.append(xrange(o, o + d))
        cnt = 0
        for tp in itertools.product(*gens):
            coord = PhyDim2(*tp)
            assert self.contains_node(coord)
            cnt += 1
            yield coord

    def rel2abs(self, rel_coordinate):
        ''' Convert relative node coordinate to absolute node coordinate. '''
        if not isinstance(rel_coordinate, PhyDim2):
            raise TypeError('NodeRegion: relative coordinate must be '
                            'a PhyDim2 object.')
        abs_coordinate = self.origin + rel_coordinate
        if not self.contains_node(abs_coordinate):
            raise ValueError('NodeRegion: relative coordinate {} is not '
                             'in node region {}'.format(rel_coordinate, self))
        return abs_coordinate

    def allocate(self, request_list):
        '''
        Allocate node subregions spatially within the node region according to
        the given `request_list`. Return a list of NodeRegion instances, or
        empty list if failed. The origin offset is absolute, not relative to
        the origin of self.

        Requests are given as a list of number of nodes.

        The strategy is to allocate stripe-wise in a zig-zag order, allowing
        for folding in width. We first determine a stripe height as the
        greatest common divisor of the requested numbers of nodes. Then
        allocate each request as (stripe height, request size / stripe height)
        to fill in the stripe, and move to the next stripe after the current
        one is filled. If the width of a request is larger than the remaining
        width of the current stripe, we use up the remaining width, and fold
        the request width to the next stripe.
        '''

        # FIXME: currently the subregions occupy the correct folded nodes in
        # the region, but the returned shapes are the unfolded ones. The
        # allocation is valid, but the subregion dimensions are not accurate,
        # so the number of hops are not not accurate.

        if sum(request_list) > self.dim.size():
            return []

        hstrp = util.gcd(self.dim.h, *request_list)
        subregions = []

        ofs_h, ofs_w = 0, 0
        move_right = True

        for req in request_list:

            # Subregion.
            assert req % hstrp == 0
            width = req // hstrp

            subdim = PhyDim2(hstrp, width)
            if move_right:
                origin = PhyDim2(ofs_h, ofs_w)
            else:
                origin = PhyDim2(ofs_h, self.dim.w - ofs_w - width)

            subregions.append(NodeRegion(dim=subdim,
                                         origin=self.origin + origin,
                                         type=self.type))

            # Move the offset
            ofs_w += width
            while ofs_w >= self.dim.w:
                # Overflow, fold to the next stripe.
                ofs_w -= self.dim.w
                ofs_h += hstrp
                move_right = not move_right

        # Not moved outside the region.
        assert ofs_h + hstrp <= self.dim.h or ofs_w == 0

        return subregions

