""" $lic$
Copyright (C) 2016-2019 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

import itertools
from collections import namedtuple

from .phy_dim2 import PhyDim2

NODE_REGION_LIST = ['dim',
                    'origin',
                    'dist',
                    'type',
                   ]

class NodeRegion(namedtuple('NodeRegion', NODE_REGION_LIST)):
    '''
    A node region defined by the dimension and origin offset.

    The `type` attribute specifies the region type, which could be `PROC` for
    computation processing nodes or 'DRAM' for off-chip data storage nodes.

    NOTE: we cannot overload __contains__ and __iter__ as a node container,
    because the base namedtuple already defines them.
    '''

    # Type enums.
    PROC = 0
    DRAM = 1
    NUM = 2

    def __new__(cls, *args, **kwargs):

        # Set default values.
        kwargs2 = kwargs.copy()
        if len(args) <= NODE_REGION_LIST.index('dist'):
            kwargs2.setdefault('dist', PhyDim2(1, 1))

        ntp = super(NodeRegion, cls).__new__(cls, *args, **kwargs2)

        if not isinstance(ntp.dim, PhyDim2):
            raise TypeError('NodeRegion: dim must be a PhyDim2 object.')
        if not isinstance(ntp.origin, PhyDim2):
            raise TypeError('NodeRegion: origin must be a PhyDim2 object.')
        if not isinstance(ntp.dist, PhyDim2):
            raise TypeError('NodeRegion: dist must be a PhyDim2 object.')

        if ntp.type not in range(cls.NUM):
            raise ValueError('NodeRegion: type must be a valid type enum.')

        return ntp

    def contains_node(self, coordinate):
        ''' Whether the region contains the given absolute node coordinate. '''
        return coordinate in self.iter_node()

    def iter_node(self):
        ''' Iterate through all absolute node coordinates in the region. '''
        for rel_coord in itertools.product(*[range(d) for d in self.dim]):
            yield self.rel2abs(PhyDim2(*rel_coord))

    def rel2abs(self, rel_coordinate):
        ''' Convert relative node coordinate to absolute node coordinate. '''
        if not isinstance(rel_coordinate, PhyDim2):
            raise TypeError('NodeRegion: relative coordinate must be '
                            'a PhyDim2 object.')
        if not all(0 <= c < d for c, d in zip(rel_coordinate, self.dim)):
            raise ValueError('NodeRegion: relative coordinate {} is not in '
                             'node region {}.'.format(rel_coordinate, self))

        abs_coordinate = self.origin + rel_coordinate * self.dist
        return abs_coordinate

