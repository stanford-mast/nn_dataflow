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
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import SchedulingConstraint, \
        SchedulingConstraintLayerPipeline

class TestSchedulingConstraintFixture(unittest.TestCase):
    ''' Base fixture class for SchedulingConstraint tests. '''

    @staticmethod
    def _gen_bl(t_end=9):
        ''' Generator for bl_t and bl_ord. '''
        return itertools.product(itertools.product(*[range(1, t_end)] * le.NUM),
                                 itertools.permutations(range(le.NUM)))


class TestSchedulingConstraint(TestSchedulingConstraintFixture):
    ''' Tests for SchedulingConstraint. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        cstr = SchedulingConstraint(topbat=2, topifm=1, topofm=4)
        self.assertEqual(cstr.topbat, 2)
        self.assertEqual(cstr.topifm, 1)
        self.assertEqual(cstr.topofm, 4)

        cstr = SchedulingConstraint(topbat=2, topofm=4)
        self.assertEqual(cstr.topbat, 2)
        self.assertEqual(cstr.topifm, 0)
        self.assertEqual(cstr.topofm, 4)

        cstr = SchedulingConstraint()
        self.assertEqual(cstr.topbat, 0)
        self.assertEqual(cstr.topifm, 0)
        self.assertEqual(cstr.topofm, 0)

    def test_invalid_args(self):
        ''' Invalid arguments. '''
        with self.assertRaisesRegexp(ValueError,
                                     'SchedulingConstraint: '
                                     '.*positive integers.*'):
            _ = SchedulingConstraint(topbat=-1, topofm=2.)

    def test_null_constraint(self):
        ''' Null constraint. '''
        cstr = SchedulingConstraint()

        self.assertTrue(cstr.is_valid_top_bl((1, 1, 2), (0, 1, 2)))
        self.assertTrue(cstr.is_valid_top_bl((3, 4, 5), (2, 1, 0)))
        self.assertTrue(cstr.is_valid_top_bl((1, 1, 1), (1, 2, 0)))

        self.assertTrue(cstr.is_valid_part(PartitionScheme(
            order=range(pe.NUM), pdims=[(2, 2)] * pe.NUM)))

    def test_is_valid_top_bl(self):
        ''' Whether is_valid_top_bl. '''
        cstr = SchedulingConstraint(topbat=2, topofm=4)
        for bl_t, bl_ord in self._gen_bl():
            valid = (bl_t[le.BAT] == 2 and bl_t[le.OFM] == 4)
            self.assertEqual(cstr.is_valid_top_bl(bl_t, bl_ord), valid)

        cstr = SchedulingConstraint(topifm=4)
        for bl_t, bl_ord in self._gen_bl():
            valid = (bl_t[le.IFM] == 4)
            self.assertEqual(cstr.is_valid_top_bl(bl_t, bl_ord), valid)

        cstr = SchedulingConstraint()
        for bl_t, bl_ord in self._gen_bl():
            self.assertTrue(cstr.is_valid_top_bl(bl_t, bl_ord))

    def test_repr(self):
        ''' __repr__. '''
        cstr = SchedulingConstraint(topbat=2)
        self.assertIn('SchedulingConstraint(', repr(cstr))
        self.assertIn('topbat=2', repr(cstr))
        self.assertIn('topifm=0', repr(cstr))
        self.assertIn('topofm=0', repr(cstr))


class TestSchedulingConstraintLayerPipeline(TestSchedulingConstraintFixture):
    ''' Tests for SchedulingConstraintLayerPipeline. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        cstr = SchedulingConstraintLayerPipeline(
            topbat=2, topifm=1, topofm=4, fbifm=True, fbofm=False)
        self.assertEqual(cstr.topbat, 2)
        self.assertEqual(cstr.topifm, 1)
        self.assertEqual(cstr.topofm, 4)

        cstr = SchedulingConstraintLayerPipeline(topbat=2, topofm=4, fbifm=True)
        self.assertEqual(cstr.topbat, 2)
        self.assertEqual(cstr.topifm, 1)
        self.assertEqual(cstr.topofm, 4)

        cstr = SchedulingConstraintLayerPipeline()
        self.assertEqual(cstr.topbat, 0)
        self.assertEqual(cstr.topifm, 0)
        self.assertEqual(cstr.topofm, 0)

        cstr = SchedulingConstraintLayerPipeline(fbifm=True, fbofm=True)
        self.assertEqual(cstr.topbat, 0)
        self.assertEqual(cstr.topifm, 1)
        self.assertEqual(cstr.topofm, 1)

    def test_invalid_args(self):
        ''' Invalid arguments. '''
        with self.assertRaisesRegexp(ValueError,
                                     'SchedulingConstraintLayerPipeline: '
                                     '.*IFM.*'):
            _ = SchedulingConstraintLayerPipeline(topifm=2, fbifm=True)

        with self.assertRaisesRegexp(ValueError,
                                     'SchedulingConstraintLayerPipeline: '
                                     '.*OFM.*'):
            _ = SchedulingConstraintLayerPipeline(topofm=2, fbofm=True)

        with self.assertRaisesRegexp(ValueError,
                                     'SchedulingConstraintLayerPipeline: '
                                     '.*IFM.*OFM.*'):
            _ = SchedulingConstraintLayerPipeline(topifm=2, topofm=2)

    def test_null_constraint(self):
        ''' Null constraint. '''
        cstr = SchedulingConstraintLayerPipeline()

        self.assertTrue(cstr.is_valid_top_bl((1, 1, 2), (0, 1, 2)))
        self.assertTrue(cstr.is_valid_top_bl((3, 4, 5), (2, 1, 0)))
        self.assertTrue(cstr.is_valid_top_bl((1, 1, 1), (1, 2, 0)))

    def test_is_valid_top_bl(self):
        ''' Whether is_valid_top_bl. '''
        cstr = SchedulingConstraintLayerPipeline(topbat=2, topofm=4, fbifm=True)
        for bl_t, bl_ord in self._gen_bl():
            valid = (bl_t[le.BAT] == 2 and bl_t[le.IFM] == 1
                     and bl_t[le.OFM] == 4
                     and bl_ord[le.BAT] > bl_ord[le.OFM])
            self.assertEqual(cstr.is_valid_top_bl(bl_t, bl_ord), valid)

        cstr = SchedulingConstraintLayerPipeline(topifm=4, fbofm=True)
        for bl_t, bl_ord in self._gen_bl():
            valid = (bl_t[le.IFM] == 4 and bl_t[le.OFM] == 1
                     and (bl_ord[le.IFM] > bl_ord[le.BAT]
                          or bl_t[le.BAT] == 1))
            self.assertEqual(cstr.is_valid_top_bl(bl_t, bl_ord), valid)

        cstr = SchedulingConstraintLayerPipeline(topofm=4)
        for bl_t, bl_ord in self._gen_bl():
            valid = (bl_t[le.OFM] == 4
                     and (bl_ord[le.OFM] > bl_ord[le.BAT]
                          or bl_t[le.BAT] == 1)
                     and (bl_ord[le.OFM] > bl_ord[le.IFM]
                          or bl_t[le.IFM] == 1))
            self.assertEqual(cstr.is_valid_top_bl(bl_t, bl_ord), valid)

        cstr = SchedulingConstraintLayerPipeline(fbifm=True)
        for bl_t, bl_ord in self._gen_bl():
            valid = (bl_t[le.IFM] == 1)
            self.assertEqual(cstr.is_valid_top_bl(bl_t, bl_ord), valid)

        cstr = SchedulingConstraintLayerPipeline()
        for bl_t, bl_ord in self._gen_bl():
            self.assertTrue(cstr.is_valid_top_bl(bl_t, bl_ord))

    def test_is_valid_part(self):
        ''' Whether is_valid_part. '''
        cstr = SchedulingConstraintLayerPipeline(
            topbat=2, topifm=1, topofm=4, fbifm=True, fbofm=False)
        self.assertTrue(cstr.is_valid_part(PartitionScheme(
            order=range(pe.NUM), pdims=[(2, 2)] * pe.NUM)))

        cstr = SchedulingConstraintLayerPipeline(topbat=2, topofm=4, fbifm=True)
        self.assertTrue(cstr.is_valid_part(PartitionScheme(
            order=range(pe.NUM), pdims=[(2, 2)] * pe.NUM)))

        cstr = SchedulingConstraintLayerPipeline()
        self.assertTrue(cstr.is_valid_part(PartitionScheme(
            order=range(pe.NUM), pdims=[(2, 2)] * pe.NUM)))

    def test_repr(self):
        ''' __repr__. '''
        cstr = SchedulingConstraintLayerPipeline(topbat=2, fbifm=True)
        self.assertIn('SchedulingConstraintLayerPipeline', repr(cstr))
        self.assertIn('topbat=2', repr(cstr))
        self.assertIn('topifm=1', repr(cstr))
        self.assertIn('topofm=0', repr(cstr))

