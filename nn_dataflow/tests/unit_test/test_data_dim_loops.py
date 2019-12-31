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

import itertools
import unittest

from nn_dataflow.core import DataDimLoops
from nn_dataflow.core import LoopEnum as le

class TestDataDimLoops(unittest.TestCase):
    ''' Tests for DataDimLoops. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        ddls = DataDimLoops(le.IFM, le.OFM)
        self.assertTupleEqual(ddls.loops(), (le.IFM, le.OFM))

        ddls = DataDimLoops(le.BAT, le.IFM, le.OFM)
        self.assertTupleEqual(ddls.loops(), (le.IFM, le.OFM, le.BAT))

    def test_valid_repeated_args(self):
        ''' Valid repeated arguments. '''
        ddls = DataDimLoops(le.IFM, le.OFM, le.IFM, le.IFM)
        self.assertTupleEqual(ddls.loops(), (le.IFM, le.OFM))

        ddls = DataDimLoops(*([le.BAT] * 10))
        self.assertTupleEqual(ddls.loops(), (le.BAT,))

    def test_invalid_args(self):
        ''' Invalid arguments. '''
        with self.assertRaisesRegex(ValueError,
                                    'DataDimLoops: .*LoopEnum.*'):
            _ = DataDimLoops(le.NUM + 1)

        with self.assertRaisesRegex(ValueError,
                                    'DataDimLoops: .*LoopEnum.*'):
            _ = DataDimLoops(le.IFM, le.NUM)

    def test_loops(self):
        ''' Get loops. '''
        for loops in self._gen_loop_combs():
            ddls = DataDimLoops(*loops)
            self.assertTupleEqual(ddls.loops(), loops)

    def test_take(self):
        ''' take. '''
        lst = [str(lpe) for lpe in range(le.NUM)]

        for loops in self._gen_loop_combs():
            ddls = DataDimLoops(*loops)
            sublst = ddls.take(lst)

            self.assertEqual(len(sublst), len(loops))
            self.assertListEqual(sublst, [str(lpe) for lpe in loops])

    def test_drop(self):
        ''' drop. '''
        lst = [str(lpe) for lpe in range(le.NUM)]

        for loops in self._gen_loop_combs():
            ddls = DataDimLoops(*loops)
            sublst = ddls.drop(lst)

            self.assertEqual(len(sublst), le.NUM - len(loops))

    def test_take_and_drop(self):
        ''' take and drop. '''
        lst = [str(lpe) for lpe in range(le.NUM)]

        for loops in self._gen_loop_combs():
            ddls = DataDimLoops(*loops)
            takelst = ddls.take(lst)
            droplst = ddls.drop(lst)

            self.assertEqual(len(takelst) + len(droplst), le.NUM)
            self.assertTrue(set(takelst).isdisjoint(set(droplst)))
            self.assertSetEqual(set(takelst) | set(droplst), set(lst))

    def test_repr(self):
        ''' __repr__. '''
        # pylint: disable=eval-used
        for loops in self._gen_loop_combs():
            ddls = DataDimLoops(*loops)
            self.assertEqual(eval(repr(ddls)), ddls)

    @staticmethod
    def _gen_loop_combs():
        ''' Generate all combinations of LoopEnum with all lengths. '''
        for num in range(1, le.NUM + 1):
            for comb in itertools.combinations(range(le.NUM), num):
                yield comb

