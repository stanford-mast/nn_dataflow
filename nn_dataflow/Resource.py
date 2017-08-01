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

from .NodeRegion import NodeRegion
from .PhyDim2 import PhyDim2

RESOURCE_LIST = ['node_region',
                 'mem_regions',
                 'dim_array',
                 'size_gbuf',
                 'size_regf',
                ]

class Resource(namedtuple('Resource', RESOURCE_LIST)):
    '''
    Hardware resource specification.

    The origins of node region and memory regions are all absolute.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(Resource, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.node_region, NodeRegion):
            raise TypeError('Resource: node_region must be '
                            'a NodeRegion instance.')

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

        if not isinstance(ntp.dim_array, PhyDim2):
            raise TypeError('Resource: dim_array must be a PhyDim2 object.')

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

