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

'''
Hardware resource specification.
'''

from collections import namedtuple

from .PhyDim2 import PhyDim2

RESOURCE_LIST = ['dim_nodes',
                 'dim_array',
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

        if hasattr(ntp.size_gbuf, '__len__'):
            raise TypeError('Cost: size_gbuf must be a scalar')
        if hasattr(ntp.size_regf, '__len__'):
            raise TypeError('Cost: size_regf must be a scalar')

        return ntp

