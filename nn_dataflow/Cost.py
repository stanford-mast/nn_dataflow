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
Cost specification.
'''

from collections import namedtuple

from . import MemHierEnum as me


class Cost(namedtuple('_Cost', ['cost_memhier', 'cost_nochop', 'cost_macop',
                                'cost_unit_static'])):
    '''
    Cost specification, including memory hierarchy cost, NoC hop cost, MAC
    operation cost, and unit-time static cost.
    '''

    def __init__(self, cost_memhier, cost_nochop, cost_macop, cost_unit_static):
        super(Cost, self).__init__(self, cost_memhier, cost_nochop, cost_macop,
                                   cost_unit_static)
        assert len(self.cost_memhier) == me.NUM
        assert not hasattr(self.cost_nochop, '__len__')  # a scalar

    def memhier(self, mhe=None):
        '''
        Return cost of memory hierarchy level `mhe`

        If None, return the list of cost for the entire hierarchy.
        '''
        return self.cost_memhier[mhe] if mhe is not None else self.cost_memhier

    def nochop(self):
        '''
        Return cost of transfer over one NoC hop.
        '''
        return self.cost_nochop

    def macop(self):
        '''
        Return cost of one MAC op.
        '''
        return self.cost_macop

    def unit_static(self):
        '''
        Return static cost of unit time.
        '''
        return self.cost_unit_static

