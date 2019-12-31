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

from . import mem_hier_enum as me

COST_LIST = ['mac_op',
             'mem_hier',
             'noc_hop',
             'idl_unit',
            ]

class Cost(namedtuple('Cost', COST_LIST)):
    '''
    Cost specification, including MAC operation cost, memory hierarchy cost,
    NoC hop cost, and idle unit-time cost.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(Cost, cls).__new__(cls, *args, **kwargs)

        if hasattr(ntp.mac_op, '__len__'):
            raise TypeError('Cost: mac_op must be a scalar')
        if not isinstance(ntp.mem_hier, tuple):
            raise TypeError('Cost: mem_hier must be a tuple')
        if len(ntp.mem_hier) != me.NUM:
            raise ValueError('Cost: mem_hier must have length {}'
                             .format(me.NUM))
        if hasattr(ntp.noc_hop, '__len__'):
            raise TypeError('Cost: noc_hop must be a scalar')
        if hasattr(ntp.idl_unit, '__len__'):
            raise TypeError('Cost: idl_unit must be a scalar')

        return ntp

    def mem_hier_at(self, mhe):
        '''
        Return cost of memory hierarchy level `mhe`.
        '''
        try:
            return self.mem_hier[mhe]
        except (IndexError, TypeError):
            return None

