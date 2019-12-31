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

from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import SchedulingConstraint, \
        SchedulingConstraintLayerPipeline

from nn_dataflow import util

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
        self.assertDictEqual(cstr.update_dict, {})

        cstr = SchedulingConstraint(topbat=2, topofm=4)
        self.assertEqual(cstr.topbat, 2)
        self.assertEqual(cstr.topifm, 0)
        self.assertEqual(cstr.topofm, 4)
        self.assertDictEqual(cstr.update_dict, {})

        cstr = SchedulingConstraint(
            topofm=4,
            update_dict={
                'l1': lambda s, _: setattr(s, 'topbat', 1),
                'l2': lambda s, r: setattr(s, 'topifm', r.topifm),
            })
        self.assertEqual(cstr.topbat, 0)
        self.assertEqual(cstr.topifm, 0)
        self.assertEqual(cstr.topofm, 4)
        self.assertEqual(len(cstr.update_dict), 2)
        self.assertIn('l1', cstr.update_dict)
        self.assertIn('l2', cstr.update_dict)

        cstr = SchedulingConstraint()
        self.assertEqual(cstr.topbat, 0)
        self.assertEqual(cstr.topifm, 0)
        self.assertEqual(cstr.topofm, 0)
        self.assertDictEqual(cstr.update_dict, {})

    def test_invalid_args(self):
        ''' Invalid arguments. '''
        with self.assertRaisesRegex(ValueError,
                                    'SchedulingConstraint: '
                                    '.*positive integers.*'):
            _ = SchedulingConstraint(topbat=-1, topofm=2.)

    def test_invalid_update_dict(self):
        ''' Invalid argument update_dict. '''
        with self.assertRaisesRegex(TypeError,
                                    'SchedulingConstraint: '
                                    '.*update_dict.*'):
            _ = SchedulingConstraint(update_dict=['l1'])

        with self.assertRaisesRegex(TypeError,
                                    'SchedulingConstraint: '
                                    '.*update_dict.*'):
            _ = SchedulingConstraint(update_dict={'l1': 1})

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

    def test_is_valid_before_update(self):
        ''' is_valid_top_bl and is_valid_part called before update. '''
        cstr = SchedulingConstraint(
            topofm=4,
            update_dict={
                'l1': lambda s, _: setattr(s, 'topbat', 1),
                'l2': lambda s, r: setattr(s, 'topifm', r.topifm),
            })

        with self.assertRaisesRegex(ValueError,
                                    'SchedulingConstraint: '
                                    '.*update_dict.*'):
            cstr.is_valid_top_bl([1] * le.NUM, range(le.NUM))

        with self.assertRaisesRegex(ValueError,
                                    'SchedulingConstraint: '
                                    '.*update_dict.*'):
            cstr.is_valid_part(PartitionScheme(order=range(pe.NUM),
                                               pdims=[(2, 2)] * pe.NUM))

    def test_filter_gen_ts(self):
        ''' Get filter_gen_ts. '''
        gen_tifm = util.factorize(36, 3)
        gen_tofm = util.factorize(20, 3)
        gen_tbat = util.factorize(16, 3)

        cstr = SchedulingConstraint(topbat=2, topofm=4)

        gifm, gifm0, gen_tifm = itertools.tee(gen_tifm, 3)
        gofm, gofm0, gen_tofm = itertools.tee(gen_tofm, 3)
        gbat, gbat0, gen_tbat = itertools.tee(gen_tbat, 3)
        fgifm, fgofm, fgbat = cstr.filter_gen_ts(gifm, gofm, gbat)

        self.assertSetEqual(set(fgifm), set(gifm0))
        set_fgofm = set(fgofm)
        set_fgbat = set(fgbat)
        self.assertTrue(set_fgofm.issubset(set(gofm0)))
        self.assertTrue(set_fgbat.issubset(set(gbat0)))
        self.assertSetEqual(set_fgofm,
                            {(4,) + tpl for tpl in util.factorize(5, 2)})
        self.assertSetEqual(set_fgbat,
                            {(2,) + tpl for tpl in util.factorize(8, 2)})

        cstr = SchedulingConstraint(topifm=4)

        gifm, gifm0, gen_tifm = itertools.tee(gen_tifm, 3)
        gofm, gofm0, gen_tofm = itertools.tee(gen_tofm, 3)
        gbat, gbat0, gen_tbat = itertools.tee(gen_tbat, 3)
        fgifm, fgofm, fgbat = cstr.filter_gen_ts(gifm, gofm, gbat)

        self.assertSetEqual(set(fgofm), set(gofm0))
        self.assertSetEqual(set(fgbat), set(gbat0))
        set_fgifm = set(fgifm)
        self.assertTrue(set_fgifm.issubset(set(gifm0)))
        self.assertSetEqual(set_fgifm,
                            {(4,) + tpl for tpl in util.factorize(9, 2)})

        cstr = SchedulingConstraint()

        gifm, gifm0, gen_tifm = itertools.tee(gen_tifm, 3)
        gofm, gofm0, gen_tofm = itertools.tee(gen_tofm, 3)
        gbat, gbat0, gen_tbat = itertools.tee(gen_tbat, 3)
        fgifm, fgofm, fgbat = cstr.filter_gen_ts(gifm, gofm, gbat)

        self.assertSetEqual(set(fgifm), set(gifm0))
        self.assertSetEqual(set(fgofm), set(gofm0))
        self.assertSetEqual(set(fgbat), set(gbat0))

    def test_update_by_prev(self):
        ''' Modifier update_by_prev. '''
        cstr = SchedulingConstraint(
            topofm=4,
            update_dict={
                'l1': lambda s, _: setattr(s, 'topbat', 1),
                'l2': lambda s, r: setattr(s, 'topifm', r.topifm),
            })
        self.assertEqual(cstr.topbat, 0)
        self.assertEqual(cstr.topifm, 0)
        self.assertEqual(cstr.topofm, 4)

        r = SchedulingConstraint(topifm=2)
        cstr.update_by_prev({'l1': None, 'l2': r})

        self.assertEqual(cstr.topbat, 1)
        self.assertEqual(cstr.topifm, 2)
        self.assertEqual(cstr.topofm, 4)

        self.assertFalse(cstr.is_valid_top_bl([1, 4, 1], range(le.NUM)))
        self.assertTrue(cstr.is_valid_top_bl([2, 4, 1], range(le.NUM)))

    def test_content_hash(self):
        ''' Content-based hash. '''
        cstr1 = SchedulingConstraint(topbat=2)
        cstr2 = SchedulingConstraint(topbat=2)
        self.assertNotEqual(id(cstr1), id(cstr2))
        self.assertEqual(hash(cstr1), hash(cstr2))
        self.assertEqual(cstr1, cstr2)

        cstr3 = SchedulingConstraint(
            topbat=2,
            update_dict={
                'l1': lambda s, _: setattr(s, 'topbat', 1),
                'l2': lambda s, r: setattr(s, 'topifm', r.topifm),
            })
        r = SchedulingConstraint(topifm=2)
        cstr3.update_by_prev({'l1': None, 'l2': r})
        cstr4 = SchedulingConstraint(topifm=2, topbat=1)
        self.assertNotEqual(id(cstr3), id(cstr4))
        self.assertEqual(hash(cstr3), hash(cstr4))
        self.assertEqual(cstr3, cstr4)

    def test_repr(self):
        ''' __repr__. '''
        cstr = SchedulingConstraint(topbat=2)
        self.assertIn('SchedulingConstraint(', repr(cstr))
        self.assertIn('topbat=2', repr(cstr))
        self.assertIn('topifm=0', repr(cstr))
        self.assertIn('topofm=0', repr(cstr))

        cstr = SchedulingConstraint(update_dict={
            'l1': lambda s, _: setattr(s, 'topbat', 1),
            'l2': lambda s, r: setattr(s, 'topifm', r.topifm),
        })
        self.assertIn('update_dict=', repr(cstr))
        self.assertIn('l1', repr(cstr))
        self.assertIn('l2', repr(cstr))


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
        with self.assertRaisesRegex(ValueError,
                                    'SchedulingConstraintLayerPipeline: '
                                    '.*IFM.*'):
            _ = SchedulingConstraintLayerPipeline(topifm=2, fbifm=True)

        with self.assertRaisesRegex(ValueError,
                                    'SchedulingConstraintLayerPipeline: '
                                    '.*OFM.*'):
            _ = SchedulingConstraintLayerPipeline(topofm=2, fbofm=True)

        with self.assertRaisesRegex(ValueError,
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

    def test_repr(self):
        ''' __repr__. '''
        cstr = SchedulingConstraintLayerPipeline(topbat=2, fbifm=True)
        self.assertIn('SchedulingConstraintLayerPipeline', repr(cstr))
        self.assertIn('topbat=2', repr(cstr))
        self.assertIn('topifm=1', repr(cstr))
        self.assertIn('topofm=0', repr(cstr))

