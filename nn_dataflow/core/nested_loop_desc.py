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

from . import data_category_enum as de
from . import loop_enum as le
from . import mem_hier_enum as me
from .. import util
from .data_dim_loops import DataDimLoops

NESTED_LOOP_DESC_LIST = ['loopcnt',
                         'usize_gbuf',
                         'usize_regf',
                         'unit_access',
                         'unit_ops',
                         'unit_time',
                         'data_loops',
                        ]

class NestedLoopDesc(namedtuple('NestedLoopDesc', NESTED_LOOP_DESC_LIST)):
    '''
    Naive nested loop description.

    For our problem, only deal with the loops given by `LoopEnum`.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(NestedLoopDesc, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.loopcnt, tuple):
            raise TypeError('NestedLoopDesc: loopcnt must be a tuple.')
        if len(ntp.loopcnt) != le.NUM:
            raise ValueError('NestedLoopDesc: loopcnt must have length {}.'
                             .format(le.NUM))

        if not isinstance(ntp.usize_gbuf, tuple):
            raise TypeError('NestedLoopDesc: usize_gbuf must be a tuple.')
        if not isinstance(ntp.usize_regf, tuple):
            raise TypeError('NestedLoopDesc: usize_regf must be a tuple.')
        if len(ntp.usize_gbuf) != de.NUM:
            raise ValueError('NestedLoopDesc: usize_gbuf must have length {}.'
                             .format(de.NUM))
        if len(ntp.usize_regf) != de.NUM:
            raise ValueError('NestedLoopDesc: usize_regf must have length {}.'
                             .format(de.NUM))

        if not isinstance(ntp.unit_access, tuple):
            raise TypeError('NestedLoopDesc: unit_access must be a tuple.')
        if len(ntp.unit_access) != me.NUM:
            raise ValueError('NestedLoopDesc: unit_access must have length {}.'
                             .format(me.NUM))
        for ua in ntp.unit_access:
            if not isinstance(ua, tuple):
                raise TypeError('NestedLoopDesc: element in unit_access '
                                'must be a tuple.')
            if len(ua) != de.NUM:
                raise ValueError('NestedLoopDesc: element in unit_access '
                                 'must have length {}.'.format(de.NUM))

        if not isinstance(ntp.data_loops, tuple):
            raise TypeError('NestedLoopDesc: data_loops must be a tuple.')
        if len(ntp.data_loops) != de.NUM:
            raise ValueError('NestedLoopDesc: data_loops must have length {}.'
                             .format(de.NUM))
        for dls in ntp.data_loops:
            if not isinstance(dls, DataDimLoops):
                raise TypeError('NestedLoopDesc: element in data_loops '
                                'must be a DataDimLoops instance.')

        return ntp

    def usize_gbuf_of(self, dce):
        '''
        Get the occupied gbuf size by data category `dce` for one loop body.
        '''
        return self.usize_gbuf[dce]

    def usize_regf_of(self, dce):
        '''
        Get the occupied regf size by data category `dce` for one loop body.
        '''
        return self.usize_regf[dce]

    def unit_access_at_of(self, mhe, dce=None):
        '''
        Get the number of accesses for one loop body at memory hierarchy `mhe`
        of data category `dce`.

        If `dce` is None, return total accesses of all data.
        '''
        if dce is None:
            return sum(self.unit_access[mhe])
        return self.unit_access[mhe][dce]

    def total_ops(self):
        '''
        Get the total number of ops for all loops.
        '''
        return self.unit_ops * util.prod(self.loopcnt)

    def total_access_at_of(self, mhe, dce=None):
        '''
        Get the total number of accesses, i.e., accessing all data once, at
        memory hierarchy `mhe` of data category `dce`.

        If `dce` is None, return total accesses of all data.
        '''
        if dce is None:
            return sum(self.total_access_at_of(mhe, dce2)
                       for dce2 in range(de.NUM))

        return self.unit_access_at_of(mhe, dce) \
                * util.prod(self.data_loops[dce].take(self.loopcnt))

