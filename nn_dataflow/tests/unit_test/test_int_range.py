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

import unittest

from nn_dataflow.core import IntRange

class TestIntRange(unittest.TestCase):
    ''' Tests for IntRange. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        ir1 = IntRange(1, 7)
        self.assertEqual(ir1.beg, 1)
        self.assertEqual(ir1.end, 7)
        ir2 = IntRange(-3, 0)
        self.assertEqual(ir2.beg, -3)
        self.assertEqual(ir2.end, 0)
        ir3 = IntRange(4, 4)
        self.assertEqual(ir3.beg, 4)
        self.assertEqual(ir3.end, 4)

    def test_invalid_args(self):
        ''' Invalid arguments. '''
        with self.assertRaisesRegex(TypeError, 'IntRange: .*beg.*'):
            _ = IntRange(7.2, 3)
        with self.assertRaisesRegex(TypeError, 'IntRange: .*end.*'):
            _ = IntRange(7, None)

        with self.assertRaisesRegex(ValueError, 'IntRange: .*beg.*end.*'):
            _ = IntRange(7, 3)
        with self.assertRaisesRegex(ValueError, 'IntRange: .*beg.*end.*'):
            _ = IntRange(-3, -7)

    def test_size(self):
        ''' Get size. '''
        ir1 = IntRange(1, 7)
        self.assertEqual(ir1.size(), 6)
        ir2 = IntRange(-3, 0)
        self.assertEqual(ir2.size(), 3)
        ir3 = IntRange(4, 4)
        self.assertEqual(ir3.size(), 0)

    def test_empty(self):
        ''' Get empty. '''
        ir1 = IntRange(1, 7)
        self.assertFalse(ir1.empty())
        ir2 = IntRange(-3, 0)
        self.assertFalse(ir2.empty())
        ir3 = IntRange(4, 4)
        self.assertTrue(ir3.empty())

    def test_range(self):
        ''' Get range. '''
        ir1 = IntRange(1, 7)
        self.assertEqual(len(set(ir1.range())), ir1.size())
        ir2 = IntRange(-3, 0)
        self.assertListEqual(list(ir2.range()), [-3, -2, -1])
        ir3 = IntRange(4, 4)
        self.assertEqual(len(list(ir3.range())), 0)

    def test_overlap(self):
        ''' Get overlap. '''
        ir1 = IntRange(-11, 5)
        ir2 = IntRange(3, 8)
        ir_ovlp = ir1.overlap(ir2)
        self.assertEqual(ir_ovlp, IntRange(3, 5))
        self.assertEqual(ir1.overlap(ir2), ir2.overlap(ir1))

        ir3 = IntRange(-3, 3)
        ir_ovlp = ir1.overlap(ir3)
        self.assertEqual(ir_ovlp, IntRange(-3, 3))

        ir4 = IntRange(8, 10)
        ir_ovlp = ir1.overlap(ir4)
        self.assertTrue(ir_ovlp.empty())

    def test_overlap_error(self):
        ''' Get overlap error. '''
        ir = IntRange(-11, 5)
        with self.assertRaisesRegex(TypeError, 'IntRange: .*'):
            ir.overlap((0, 1))

    def test_offset(self):
        ''' Get offset. '''
        ir1 = IntRange(1, 7)
        self.assertEqual(ir1.offset(3), IntRange(4, 10))
        ir2 = IntRange(-3, 0)
        self.assertEqual(ir2.offset(-2), IntRange(-5, -2))

