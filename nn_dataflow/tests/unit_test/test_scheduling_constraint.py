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

import itertools
import unittest

from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import SchedulingConstraint

class TestSchedulingConstraint(unittest.TestCase):
    ''' Tests for SchedulingConstraint. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        constraint = SchedulingConstraint(top_bl_t=(1, 1, 2),
                                          top_bl_lpe=le.BAT)
        self.assertEqual(constraint.top_bl_t, (1, 1, 2))
        self.assertEqual(constraint.top_bl_lpe, le.BAT)

        constraint = SchedulingConstraint(top_bl_t=(1, 1, 2))
        self.assertIsNone(constraint.top_bl_lpe)

        constraint = SchedulingConstraint(top_bl_t=(1, None, None))
        self.assertEqual(constraint.top_bl_t, (1, None, None))
        self.assertIsNone(constraint.top_bl_lpe)

        constraint = SchedulingConstraint(top_bl_lpe=le.BAT)
        self.assertIsNone(constraint.top_bl_t)

        constraint = SchedulingConstraint()
        self.assertIsNone(constraint.top_bl_t)
        self.assertIsNone(constraint.top_bl_lpe)

    def test_invalid_top_bl_t(self):
        ''' Invalid top_bl_t. '''
        with self.assertRaisesRegexp(TypeError,
                                     'SchedulingConstraint: .*top_bl_t.*'):
            _ = SchedulingConstraint(top_bl_t=[1, 1, 2])

        with self.assertRaisesRegexp(ValueError,
                                     'SchedulingConstraint: .*top_bl_t.*'):
            _ = SchedulingConstraint(top_bl_t=(1, 2))

    def test_invalid_top_bl_lpe(self):
        ''' Invalid top_bl_lpe. '''
        with self.assertRaisesRegexp(ValueError,
                                     'SchedulingConstraint: .*top_bl_lpe.*'):
            _ = SchedulingConstraint(top_bl_lpe=le.NUM)

    def test_is_valid_top_bl_t(self):
        ''' Whether is_valid_top_bl for top_bl_t. '''
        top_bl_ord = range(le.NUM)

        constraint = SchedulingConstraint(top_bl_t=(1, 1, 2))
        self.assertTrue(constraint.is_valid_top_bl((1, 1, 2), top_bl_ord))
        self.assertFalse(constraint.is_valid_top_bl((1, 2, 2), top_bl_ord))
        self.assertFalse(constraint.is_valid_top_bl((1, 2, 1), top_bl_ord))
        self.assertFalse(constraint.is_valid_top_bl((2, 2, 2), top_bl_ord))

        constraint = SchedulingConstraint(top_bl_t=(1, None, 3))
        for t1 in range(10):
            self.assertTrue(constraint.is_valid_top_bl((1, t1, 3), top_bl_ord))

        constraint = SchedulingConstraint()
        for top_bl_t in itertools.product(range(5), range(5), range(5)):
            self.assertTrue(constraint.is_valid_top_bl(top_bl_t, top_bl_ord))

        # Multiple of le.BAT factor.
        constraint = SchedulingConstraint(top_bl_t=(None, None, 3))
        for t2 in range(3, 21, 3):
            self.assertTrue(constraint.is_valid_top_bl((1, 1, t2), top_bl_ord))

    def test_is_valid_top_bl_ord(self):
        ''' Whether is_valid_top_bl for top_bl_ord. '''
        top_bl_t = [2] * le.NUM

        constraint = SchedulingConstraint()
        for top_bl_ord in itertools.permutations(range(le.NUM)):
            self.assertTrue(constraint.is_valid_top_bl(top_bl_t, top_bl_ord))

        constraint = SchedulingConstraint(top_bl_lpe=le.BAT)
        self.assertTrue(constraint.is_valid_top_bl(top_bl_t, (0, 1, 2)))
        self.assertTrue(constraint.is_valid_top_bl(top_bl_t, (1, 0, 2)))
        self.assertFalse(constraint.is_valid_top_bl(top_bl_t, (0, 2, 1)))
        self.assertFalse(constraint.is_valid_top_bl(top_bl_t, (2, 0, 1)))
        self.assertFalse(constraint.is_valid_top_bl(top_bl_t, (1, 2, 0)))
        self.assertFalse(constraint.is_valid_top_bl(top_bl_t, (2, 1, 0)))

        top_bl_t[le.IFM] = 1
        self.assertTrue(constraint.is_valid_top_bl(top_bl_t, (2, 0, 1)))

        top_bl_t[le.OFM] = 1
        for top_bl_ord in itertools.permutations(range(le.NUM)):
            self.assertTrue(constraint.is_valid_top_bl(top_bl_t, top_bl_ord))

        top_bl_t = (1, 1, 1)
        for top_bl_ord in itertools.permutations(range(le.NUM)):
            self.assertTrue(constraint.is_valid_top_bl(top_bl_t, top_bl_ord))

