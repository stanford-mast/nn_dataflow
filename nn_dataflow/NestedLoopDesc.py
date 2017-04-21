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

from . import DataCategoryEnum as de
from . import MemHierEnum as me

NESTED_LOOP_DESC_LIST = ['loopcnt_ifm',
                         'loopcnt_ofm',
                         'loopcnt_bat',
                         'usize_gbuf',
                         'usize_regf',
                         'unit_access',
                         'unit_ops',
                         'unit_time',
                        ]

class NestedLoopDesc(namedtuple('NestedLoopDesc', NESTED_LOOP_DESC_LIST)):
    '''
    Naive 3-nested loop (nifm, nofm, batch) description.

    For our problem, only deal with nifm, nofm, and batch loops.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(NestedLoopDesc, cls).__new__(cls, *args, **kwargs)

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

