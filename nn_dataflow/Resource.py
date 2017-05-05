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

from .PhyDim2 import PhyDim2

NODE_REGION_LIST = ['dim',
                    'origin',
                   ]

class NodeRegion(namedtuple('NodeRegion', NODE_REGION_LIST)):
    '''
    A node region defined by the dimension and origin offset.

    NOTES: we cannot overload __contains__ and __iter__ as a node container,
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
            raise TypeError('Cost: size_gbuf must be a scalar')
        if hasattr(ntp.size_regf, '__len__'):
            raise TypeError('Cost: size_regf must be a scalar')

        return ntp

    def mem_region_src(self):
        ''' Get the memory region for the data source. '''
        return self.mem_regions[0]

    def mem_region_dst(self):
        ''' Get the memory region for the data destination. '''
        return self.mem_regions[-1]

