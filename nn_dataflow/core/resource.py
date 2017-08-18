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

from .node_region import NodeRegion
from .phy_dim2 import PhyDim2

RESOURCE_LIST = ['proc_region',
                 'data_regions',
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

        if not isinstance(ntp.proc_region, NodeRegion):
            raise TypeError('Resource: proc_region must be '
                            'a NodeRegion instance.')
        if ntp.proc_region.type != NodeRegion.PROC:
            raise ValueError('Resource: proc_region must have type PROC.')

        if not isinstance(ntp.data_regions, tuple):
            raise TypeError('Resource: data_regions must be a tuple.')
        for dr in ntp.data_regions:
            if not isinstance(dr, NodeRegion):
                raise TypeError('Resource: element in data_regions must be '
                                'a NodeRegion instance.')
            if dr.type != NodeRegion.DATA:
                raise ValueError('Resource: element in data_regions must have '
                                 'type DATA.')
        # Data regions can be used as either data source or data destination.
        # If a single region is provided, it is both the source and
        # destination; if two regions are provided, the first is the source and
        # the second is the destination.
        if len(ntp.data_regions) > 2:
            raise ValueError('Resource: can have at most 2 data_regions.')

        if not isinstance(ntp.dim_array, PhyDim2):
            raise TypeError('Resource: dim_array must be a PhyDim2 object.')

        if hasattr(ntp.size_gbuf, '__len__'):
            raise TypeError('Resource: size_gbuf must be a scalar')
        if hasattr(ntp.size_regf, '__len__'):
            raise TypeError('Resource: size_regf must be a scalar')

        return ntp

    def src_data_region(self):
        ''' Get the source data region. '''
        return self.data_regions[0]

    def dst_data_region(self):
        ''' Get the destination data region. '''
        return self.data_regions[-1]

