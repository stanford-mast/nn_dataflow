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
import math

from .node_region import NodeRegion
from .phy_dim2 import PhyDim2

RESOURCE_LIST = ['proc_region',
                 'dram_region',
                 'src_data_region',
                 'dst_data_region',
                 'dim_array',
                 'size_gbuf',
                 'size_regf',
                 'array_bus_width',
                 'dram_bandwidth',
                 'no_time_mux',
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

        if not isinstance(ntp.dram_region, NodeRegion):
            raise TypeError('Resource: dram_region must be '
                            'a NodeRegion instance.')
        if ntp.dram_region.type != NodeRegion.DRAM:
            raise ValueError('Resource: dram_region must have type DRAM.')

        if not isinstance(ntp.src_data_region, NodeRegion):
            raise TypeError('Resource: src_data_region must be '
                            'a NodeRegion instance.')
        if not isinstance(ntp.dst_data_region, NodeRegion):
            raise TypeError('Resource: dst_data_region must be '
                            'a NodeRegion instance.')

        if not isinstance(ntp.dim_array, PhyDim2):
            raise TypeError('Resource: dim_array must be a PhyDim2 object.')

        if hasattr(ntp.size_gbuf, '__len__'):
            raise TypeError('Resource: size_gbuf must be a scalar')
        if hasattr(ntp.size_regf, '__len__'):
            raise TypeError('Resource: size_regf must be a scalar')

        if not isinstance(ntp.array_bus_width, int) \
                and not math.isinf(ntp.array_bus_width):
            raise TypeError('Resource: array_bus_width must be an integer '
                            'or infinity.')
        if ntp.array_bus_width <= 0:
            raise ValueError('Resource: array_bus_width must be positive.')

        if not isinstance(ntp.dram_bandwidth, (float, int)):
            raise TypeError('Resource: dram_bandwidth must be a number')
        if ntp.dram_bandwidth <= 0:
            raise ValueError('Resource: dram_bandwidth must be positive.')

        if not isinstance(ntp.no_time_mux, bool):
            raise TypeError('Resource: no_time_mux must be boolean')

        return ntp

