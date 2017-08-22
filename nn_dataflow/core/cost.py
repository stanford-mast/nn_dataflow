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

from . import mem_hier_enum as me

COST_LIST = ['mac_op',
             'mem_hier',
             'noc_hop',
             'unit_static'
            ]

class Cost(namedtuple('Cost', COST_LIST)):
    '''
    Cost specification, including MAC operation cost, memory hierarchy cost,
    NoC hop cost, and unit-time static cost.
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
        if hasattr(ntp.unit_static, '__len__'):
            raise TypeError('Cost: unit_static must be a scalar')

        return ntp

    def mem_hier_at(self, mhe):
        '''
        Return cost of memory hierarchy level `mhe`.
        '''
        try:
            return self.mem_hier[mhe]
        except (IndexError, TypeError):
            return None

