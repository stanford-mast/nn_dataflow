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

from nn_dataflow.core import FmapPosition
from nn_dataflow.core import FmapRange
from nn_dataflow.core import FmapRangeMap

class TestFmapRange(unittest.TestCase):
    ''' Tests for FmapRange. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        fr = FmapRange((0, 0, 0, 0), (3, 5, 7, 11))
        self.assertTupleEqual(fr.fp_beg, (0, 0, 0, 0), 'fp_beg')
        self.assertTupleEqual(fr.fp_end, (3, 5, 7, 11), 'fp_end')

    def test_invalid_beg_end(self):
        ''' Invalid fp_beg/fp_end. '''
        with self.assertRaisesRegex(ValueError, 'FmapRange: .*beg.*end.*'):
            _ = FmapRange((0, 0, 0, 0), (3, -5, 7, 11))

    def test_valid_zero_range(self):
        ''' Valid zero range, i.e., fp_beg == fp_end. '''
        fr = FmapRange((0, 1, 2, 3), (2, 4, 2, 5))
        self.assertEqual(fr.size(), 0, 'Zero range')

    def test_beg_end(self):
        ''' Get beg_end. '''
        fr = FmapRange((-11, -4, 3, 0), (3, 5, 7, 11))

        be = fr.beg_end('n')
        self.assertTupleEqual(be, (-4, 5), 'beg_end: n: val')

        be = fr.beg_end('h')
        self.assertTupleEqual(be, (3, 7), 'beg_end: h: val')

        be_b, be_w = fr.beg_end('b', 'w')
        self.assertTupleEqual(be_b, (-11, 3), 'beg_end: b: val')
        self.assertTupleEqual(be_w, (0, 11), 'beg_end: w: val')

        be_all = fr.beg_end()
        self.assertEqual(len(be_all), 4, 'beg_end: all: len')
        self.assertTupleEqual(be_all[0], (-11, 3), 'beg_end: all: b: val')
        self.assertTupleEqual(be_all[1], (-4, 5), 'beg_end: all: n: val')
        self.assertTupleEqual(be_all[2], (3, 7), 'beg_end: all: h: val')
        self.assertTupleEqual(be_all[3], (0, 11), 'beg_end: all: w: val')

    def test_range(self):
        ''' Get range. '''
        fr = FmapRange((-11, -4, 3, 0), (3, 5, 7, 11))

        self.assertEqual(len(set(fr.range('n', 'w'))), fr.size('n', 'w'),
                         'range: nw: len')
        self.assertEqual(len(set(fr.range('h'))), fr.size('h'), 'range: h: len')
        self.assertEqual(len(set(fr.range())), fr.size(), 'range: all: len')

        be_b, be_w = fr.beg_end('b', 'w')
        for pnt in fr.range('b', 'w'):
            self.assertGreaterEqual(pnt[0], be_b[0], 'range: b: >=')
            self.assertLess(pnt[0], be_b[1], 'range: b: <')
            self.assertGreaterEqual(pnt[1], be_w[0], 'range: w: >=')
            self.assertLess(pnt[1], be_w[1], 'range: w: <')

        for fp in fr.range():
            self.assertTrue(fp in fr, 'range: all: in')

    def test_size(self):
        ''' Get size. '''
        fr = FmapRange((-11, -4, 3, 0), (3, 5, 7, 11))

        self.assertEqual(fr.size('n', 'w'), (5 + 4) * 11, 'size: nw')
        self.assertEqual(fr.size('h'), 7 - 3, 'size: h')
        self.assertEqual(fr.size(), 14 * 9 * 4 * 11, 'size: all')

    def test_overlap(self):
        ''' Get overlap. '''
        fr1 = FmapRange((-11, -4, 3, 0), (3, 5, 7, 11))
        fr2 = FmapRange((0, 3, 3, -5), (3, 10, 4, 3))
        ofr = fr1.overlap(fr2)
        self.assertListEqual(ofr.beg_end(),
                             [(0, 3), (3, 5), (3, 4), (0, 3)],
                             'overlap')
        self.assertEqual(fr2.overlap(fr1), ofr, 'overlap: commutative')

        fr3 = FmapRange((0, 7, 3, -5), (3, 10, 4, 3))
        ofr = fr1.overlap(fr3)
        self.assertListEqual(ofr.beg_end(), [(0, 0)] * 4, 'overlap: no')

        fr4 = FmapRange((-12, -12, -12, -12), (12, 12, 12, 12))
        self.assertEqual(fr1.overlap(fr4), fr1)

    def test_overlap_error(self):
        ''' Get overlap error. '''
        fr1 = FmapRange((-11, -4, 3, 0), (3, 5, 7, 11))
        with self.assertRaisesRegex(TypeError, 'FmapRange: .*'):
            fr1.overlap(((0,) * 4, (2,) * 4))

    def test_overlap_size(self):
        ''' Get overlap_size. '''
        fr1 = FmapRange((-11, -4, 3, 0), (3, 5, 7, 11))
        fr2 = FmapRange((0, 3, 3, -5), (3, 10, 4, 3))
        self.assertEqual(fr1.overlap_size(fr2), 3 * 2 * 1 * 3)
        self.assertEqual(fr2.overlap_size(fr1), 3 * 2 * 1 * 3)

        fr3 = FmapRange((0, 7, 3, -5), (3, 10, 4, 3))
        self.assertEqual(fr1.overlap_size(fr3), 0)

        fr4 = FmapRange((-12, -12, -12, -12), (12, 12, 12, 12))
        self.assertEqual(fr1.overlap_size(fr4), fr1.size())

    def test_overlap_size_error(self):
        ''' Get overlap_size error. '''
        fr1 = FmapRange((-11, -4, 3, 0), (3, 5, 7, 11))
        with self.assertRaisesRegex(TypeError, 'FmapRange: .*'):
            fr1.overlap_size(((0,) * 4, (2,) * 4))

    def test_contains(self):
        ''' Whether contains fmap point. '''
        fr = FmapRange((-11, -4, 3, 0), (3, 5, 7, 11))

        for fp in fr.range():
            self.assertTrue(fp in fr, 'contains')

        num = 0
        for fp in FmapRange((-12, -12, -12, -12), (12, 12, 12, 12)).range():
            num += 1 if fp in fr else 0
        self.assertEqual(num, fr.size())

    def test_compare(self):
        ''' Comparison. '''
        # Create non-overlapping FmapRange instances.
        lst = []
        for b, n, h, w in itertools.product(*[range(5) for _ in range(4)]):
            lst.append(FmapRange((b, n, h, w), (b + 1, n + 1, h + 1, w + 1)))

        # Sort.
        lst = sorted(lst)

        for idx in range(len(lst) - 1):
            self.assertLess(lst[idx], lst[idx + 1])
            self.assertGreater(lst[idx + 1], lst[idx])
            self.assertLessEqual(lst[idx], lst[idx])
            self.assertGreaterEqual(lst[idx], lst[idx])

    def test_compare_overlap(self):
        ''' Comparison with overlapping FmapRange. '''
        fr1 = FmapRange((-11, -4, 3, 0), (3, 5, 7, 11))
        fr2 = FmapRange((-11, -4, 3, 0), (1, 1, 5, 5))
        fr3 = FmapRange((0, 0, 3, 1), (1, 1, 5, 5))
        fr4 = FmapRange((0, 0, 3, 1), (3, 5, 7, 11))
        with self.assertRaisesRegex(ValueError, 'FmapRange: .*overlap.*'):
            _ = fr1 < fr2
        with self.assertRaisesRegex(ValueError, 'FmapRange: .*overlap.*'):
            _ = fr1 >= fr2
        with self.assertRaisesRegex(ValueError, 'FmapRange: .*overlap.*'):
            _ = fr1 <= fr3
        with self.assertRaisesRegex(ValueError, 'FmapRange: .*overlap.*'):
            _ = fr1 < fr4

    def test_compare_empty(self):
        ''' Comparison with empty FmapRange. '''
        fr1 = FmapRange((-11, -4, 3, 0), (3, 5, 7, 11))
        fr2 = FmapRange((100, 1, 1, 1), (101, 2, 2, 2))
        fr3 = FmapRange((1, 1, 1, 1), (1, 2, 2, 2))
        fr4 = FmapRange((0, 0, 0, 0), (0, 0, 0, 0))
        self.assertLess(fr1, fr2)
        self.assertGreater(fr1, fr3)

        self.assertTrue(fr3 == fr4 < fr1 < fr2)

    def test_eq(self):
        ''' Whether eq. '''
        fr1 = FmapRange((1, 2, 3, 4), (5, 7, 11, 13))
        fr2 = FmapRange((1, 2, 3, 4), (5, 7, 11, 13))
        self.assertEqual(fr1, fr2)

        fr3 = FmapRange((1, 1, 1, 1), (1, 2, 2, 2))
        fr4 = FmapRange((0, 0, 0, 0), (0, 0, 0, 0))
        self.assertEqual(fr3, fr4)

    def test_ne(self):
        ''' Whether ne. '''
        fr1 = FmapRange((1, 2, 3, 4), (5, 7, 11, 13))
        fr2 = FmapRange((1, 2, 3, 4), (5, 7, 11, 17))
        fr3 = FmapRange((1, 2, 3, 17), (5, 7, 11, 20))
        self.assertNotEqual(fr1, fr2)
        self.assertNotEqual(fr1, fr3)

    def test_hash(self):
        ''' Get hash. '''
        fr1 = FmapRange((1, 2, 3, 4), (5, 7, 11, 13))
        fr2 = FmapRange((1, 2, 3, 4), (5, 7, 11, 13))
        self.assertEqual(hash(fr1), hash(fr2))

    def test_repr(self):
        ''' __repr__. '''
        # pylint: disable=eval-used
        for fr in [FmapRange((0, 0, 0, 0), (3, 5, 7, 11)),
                   FmapRange((-11, -4, 3, 0), (3, 5, 7, 11)),
                   FmapRange((-11, -4, 3, 0), (3, 5, 7, 11)),
                   FmapRange((0, 0, 3, 1), (1, 1, 5, 5))]:
            self.assertEqual(eval(repr(fr)), fr)


class TestFmapRangeMap(unittest.TestCase):
    ''' Tests for FmapRangeMap. '''

    def setUp(self):
        self.frm = FmapRangeMap()
        self.frm.add(FmapRange((0, 0, 0, 0), (2, 4, 8, 16)), 0)
        self.frm.add(FmapRange((0, 0, 8, 0), (2, 4, 16, 16)), 1)
        self.frm.add(FmapRange((0, 4, 0, 0), (2, 8, 8, 16)), 2)
        self.frm.add(FmapRange((0, 4, 8, 0), (2, 8, 16, 16)), 3)
        self.frm.add(FmapRange((2, 0, 0, 0), (4, 4, 8, 16)), 4)
        self.frm.add(FmapRange((2, 0, 8, 0), (4, 4, 16, 16)), 5)
        self.frm.add(FmapRange((2, 4, 0, 0), (4, 8, 8, 16)), 6)
        self.frm.add(FmapRange((2, 4, 8, 0), (4, 8, 16, 16)), 7)

    def test_add(self):
        ''' Modifier add. '''
        self.frm.add(FmapRange((4, 8, 16, 16), (5, 9, 17, 17)), 10)
        self.assertEqual(self.frm.get(FmapPosition(4, 8, 16, 16)), 10, 'add')
        self.frm.add(FmapRange((10, 10, 20, 20), (15, 19, 27, 27)), 11)
        self.assertEqual(self.frm.get(FmapPosition(14, 15, 22, 24)), 11, 'add')

    def test_add_zero_fr(self):
        ''' Modifier add zero FmapRange. '''
        num_items = len(list(self.frm.items()))
        self.frm.add(FmapRange((5, 9, 17, 17), (5, 9, 17, 17)), 10)
        self.assertEqual(len(list(self.frm.items())), num_items)

    def test_add_overlap_fr(self):
        ''' Modifier add overlapping FmapRange. '''
        with self.assertRaisesRegex(ValueError, 'FmapRangeMap: .*overlap.*'):
            self.frm.add(FmapRange((3, 7, 15, 15), (5, 9, 17, 17)), 10)

    def test_get(self):
        ''' Get. '''
        self.assertEqual(self.frm.get(FmapPosition(3, 5, 7, 9)), 6, 'get')
        self.assertEqual(self.frm.get(FmapPosition(0, 0, 0, 0)), 0, 'get')
        self.assertEqual(self.frm.get(FmapPosition(2, 1, 1, 12)), 4, 'get')
        self.assertEqual(self.frm.get(FmapPosition(3, 7, 15, 15)), 7, 'get')

    def test_get_not_in(self):
        ''' Get not in. '''
        with self.assertRaisesRegex(KeyError, 'FmapRangeMap: .*key.*'):
            _ = self.frm.get(FmapPosition(4, 8, 16, 16))

    def test_complete_fmap_range(self):
        ''' Get complete_fmap_range. '''
        self.assertTrue(self.frm.is_complete(), 'is_complete')
        self.assertEqual(self.frm.complete_fmap_range().size(), 4 * 8 * 16 * 16,
                         'complete_fmap_range')

        fr = FmapRange((0, 0, 0, 0), (3, 5, 7, 9))
        frm = FmapRangeMap()
        frm.add(fr, 3.4)
        self.assertTrue(frm.is_complete(), 'is_complete')
        self.assertEqual(frm.complete_fmap_range(), fr, 'complete_fmap_range')

    def test_is_complete_incomplete(self):
        ''' Get complete_fmap_range incomplete. '''
        self.frm.add(FmapRange((4, 8, 16, 16), (5, 9, 17, 17)), 10)
        self.assertFalse(self.frm.is_complete(), 'is_complete: incomplete')
        with self.assertRaisesRegex(ValueError, 'FmapRangeMap: .*complete.*'):
            _ = self.frm.complete_fmap_range()

        fr = FmapRange((1, 0, 0, 0), (3, 5, 7, 9))
        frm = FmapRangeMap()
        frm.add(fr, 3.4)
        self.assertFalse(frm.is_complete(), 'is_complete: incomplete')
        with self.assertRaisesRegex(ValueError, 'FmapRangeMap: .*complete.*'):
            _ = frm.complete_fmap_range()

    def test_items(self):
        ''' Accessor items. '''
        size = 0
        for k, v in self.frm.items():
            size += k.size()
            self.assertEqual(self.frm.get(k.fp_beg), v, 'items: keyval')
        self.assertEqual(size, 4 * 8 * 16 * 16, 'items: size')

    def test_copy(self):
        ''' Copy. '''
        frm = self.frm.copy()
        self.assertListEqual(list(frm.items()), list(self.frm.items()),
                             'copy: equal')

        fr1 = FmapRange((10, 10, 10, 10), (11, 11, 11, 11))
        frm.add(fr1, 10)
        self.assertEqual(frm.get(fr1.fp_beg), 10, 'copy: in')
        with self.assertRaisesRegex(KeyError, 'FmapRangeMap: .*key.*'):
            _ = self.frm.get(fr1.fp_beg)

        fr2 = FmapRange((20, 20, 20, 20), (21, 21, 21, 21))
        self.frm.add(fr2, 20)
        self.assertEqual(self.frm.get(fr2.fp_beg), 20, 'copy: in')
        with self.assertRaisesRegex(KeyError, 'FmapRangeMap: .*key.*'):
            _ = frm.get(fr2.fp_beg)

    def test_rget_counter(self):
        ''' Get rget_counter. '''
        fr = FmapRange((1, 3, 9, 11), (3, 5, 13, 15))
        counters = self.frm.rget_counter(fr)
        self.assertEqual(sum(counters.values()), fr.size(), 'rget_counter')

        fr = FmapRange((0, 0, 0, 0), (0, 0, 0, 0))
        counters = self.frm.rget_counter(fr)
        self.assertEqual(sum(counters.values()), 0, 'rget_counter')

        fr = FmapRange((1, 3, 9, 11), (3, 5, 13, 17))
        counters = self.frm.rget_counter(fr)
        self.assertLess(sum(counters.values()), fr.size(), 'rget_counter')
        self.assertEqual(sum(counters.values()),
                         self.frm.complete_fmap_range().overlap(fr).size(),
                         'rget_counter')

    def test_rget_counter_same_vals(self):
        ''' Get rget_counter when there are same values in FmapRangeMap. '''
        self.frm.add(FmapRange((0, 0, 0, 16), (4, 8, 16, 32)), 2)
        fr = FmapRange((1, 3, 9, 11), (3, 5, 13, 17))
        counters = self.frm.rget_counter(fr)
        self.assertEqual(sum(counters.values()), fr.size())

    def test_rget_single(self):
        ''' Get rget_single. '''
        for k, v in self.frm.items():
            self.assertEqual(self.frm.rget_single(k), v, 'rget_single')

        val = self.frm.rget_single(FmapRange((0, 0, 0, 0), (1, 1, 1, 1)))
        self.assertEqual(val, 0, 'rget_single')

        val = self.frm.rget_single(FmapRange((3, 1, 10, 3), (4, 3, 13, 7)))
        self.assertEqual(val, 5, 'rget_single')

    def test_rget_single_multi(self):
        ''' Get rget_single with . '''
        with self.assertRaisesRegex(ValueError, 'FmapRangeMap: .*single.*'):
            _ = self.frm.rget_single(FmapRange((3, 1, 10, 3), (4, 6, 13, 7)))

    def test_str(self):
        ''' Get string. '''
        string = str(self.frm)
        for k, v in self.frm.items():
            self.assertIn(str(k), string)
            self.assertIn(str(v), string)

