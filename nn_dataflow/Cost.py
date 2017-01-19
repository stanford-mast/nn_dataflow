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

