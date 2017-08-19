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

from . import loop_enum as le
from .. import util

class DataDimLoops(util.ContentHashClass):
    '''
    A tuple of loops that are the dimensions of the data.
    '''

    def __init__(self, *lpe_list):
        for lpe in lpe_list:
            if lpe not in range(le.NUM):
                raise ValueError('DataDimLoops: arguments must be LoopEnum.')

        self.lpe_tuple = tuple(sorted(set(lpe_list)))

    def loops(self):
        '''
        Get the loops that are the dimensions of the data.
        '''
        return self.lpe_tuple

    def take(self, lpe_indexed):
        '''
        Get the elements in `lpe_indexed` that correspond to the loops of the
        data.
        '''
        return [lpe_indexed[lpe] for lpe in self.lpe_tuple]

    def drop(self, lpe_indexed):
        '''
        Get the elements in `lpe_indexed` that do not correspond to the loops
        of the data.
        '''
        return [lpe_indexed[lpe] for lpe in range(le.NUM)
                if lpe not in self.lpe_tuple]

    def data_cnt(self, loop_cnt):
        '''
        Get the data unit count from the given loop counts.
        '''
        return util.prod(self.take(loop_cnt))

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([repr(lpe) for lpe in self.lpe_tuple]))

