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

    NOTES: we cannot overload __contains__ and __iter__ as a node container,
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

