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

from . import Util
from .PhyDim2 import PhyDim2

NODE_REGION_LIST = ['dim',
                    'origin',
                   ]

class NodeRegion(namedtuple('NodeRegion', NODE_REGION_LIST)):
    '''
    A node region defined by the dimension and origin offset.

    NOTE: we cannot overload __contains__ and __iter__ as a node container,
    because the base namedtuple already defines them.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(NodeRegion, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.dim, PhyDim2):
            raise TypeError('NodeRegion: dim must be a PhyDim2 object.')
        if not isinstance(ntp.origin, PhyDim2):
            raise TypeError('NodeRegion: origin must be a PhyDim2 object.')

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

        hstrp = Util.gcd(self.dim.h, *request_list)
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
                                         origin=self.origin + origin))

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


RESOURCE_LIST = ['dim_nodes',
                 'dim_array',
                 'mem_regions',
                 'size_gbuf',
                 'size_regf',
                ]

class Resource(namedtuple('Resource', RESOURCE_LIST)):
    '''
    Hardware resource specification.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(Resource, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.dim_nodes, PhyDim2):
            raise TypeError('Resource: dim_nodes must be a PhyDim2 object.')
        if not isinstance(ntp.dim_array, PhyDim2):
            raise TypeError('Resource: dim_array must be a PhyDim2 object.')

        if not isinstance(ntp.mem_regions, tuple):
            raise TypeError('Resource: mem_regions must be a tuple.')
        for mr in ntp.mem_regions:
            if not isinstance(mr, NodeRegion):
                raise TypeError('Resource: element in mem_regions must be '
                                'a NodeRegion instance.')
        # Memory regions can be used as either data source or data destination.
        # If a single region is provided, it is both the source and
        # destination; if two regions are provided, the first is the source and
        # the second is the destination.
        if len(ntp.mem_regions) > 2:
            raise ValueError('Resource: can have at most 2 mem_regions.')

        if hasattr(ntp.size_gbuf, '__len__'):
            raise TypeError('Resource: size_gbuf must be a scalar')
        if hasattr(ntp.size_regf, '__len__'):
            raise TypeError('Resource: size_regf must be a scalar')

        return ntp

    def mem_region_src(self):
        ''' Get the memory region for the data source. '''
        return self.mem_regions[0]

    def mem_region_dst(self):
        ''' Get the memory region for the data destination. '''
        return self.mem_regions[-1]

