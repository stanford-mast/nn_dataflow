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

import unittest

from nn_dataflow.core import DataLayout
from nn_dataflow.core import FmapRange, FmapRangeMap
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2

class TestDataLayout(unittest.TestCase):
    ''' Tests for DataLayout. '''

    def setUp(self):
        self.frm = FmapRangeMap()
        self.frm.add(FmapRange((0, 0, 0, 0), (2, 4, 8, 16)), (PhyDim2(0, 0),))
        self.frm.add(FmapRange((0, 0, 8, 0), (2, 4, 16, 16)), (PhyDim2(1, 0),))
        self.frm.add(FmapRange((2, 0, 0, 0), (4, 4, 8, 16)), (PhyDim2(0, 1),))
        self.frm.add(FmapRange((2, 0, 8, 0), (4, 4, 16, 16)), (PhyDim2(1, 1),))
        self.dly = DataLayout(origin=PhyDim2(1, 1), frmap=self.frm,
                              type=NodeRegion.DATA)

    def test_valid_args(self):
        ''' Valid arguments. '''
        self.assertEqual(self.dly.frmap.complete_fmap_range(),
                         FmapRange((0, 0, 0, 0), (4, 4, 16, 16)))

    def test_invalid_origin(self):
        ''' Invalid origin. '''
        with self.assertRaisesRegexp(TypeError, 'DataLayout: .*origin.*'):
            _ = DataLayout(origin=(2, 3), frmap=self.frm, type=NodeRegion.DATA)

    def test_invalid_frmap_type(self):
        ''' Invalid frmap type. '''
        with self.assertRaisesRegexp(TypeError, 'DataLayout: .*frmap.*'):
            _ = DataLayout(origin=PhyDim2(2, 3),
                           frmap=FmapRange((0,) * 4, (1,) * 4),
                           type=NodeRegion.DATA)

    def test_invalid_frmap_incomplete(self):
        ''' Invalid frmap incomplete. '''
        frm = self.frm.copy()
        frm.add(FmapRange((4, 4, 16, 16), (5, 5, 19, 19)), (PhyDim2(4, 4),))
        with self.assertRaisesRegexp(ValueError, 'DataLayout: .*frmap.*'):
            _ = DataLayout(origin=PhyDim2(2, 3), frmap=frm,
                           type=NodeRegion.DATA)

    def test_invalid_frmap_value_type(self):
        ''' Invalid frmap value type. '''
        frm = FmapRangeMap()
        frm.add(FmapRange((0,) * 4, (5, 5, 19, 19)), PhyDim2(4, 4))
        with self.assertRaisesRegexp(TypeError, 'DataLayout: .*frmap.*'):
            _ = DataLayout(origin=PhyDim2(2, 3), frmap=frm,
                           type=NodeRegion.DATA)
        frm = FmapRangeMap()
        frm.add(FmapRange((0,) * 4, (5, 5, 19, 19)), (PhyDim2(4, 4), 4))
        with self.assertRaisesRegexp(TypeError, 'DataLayout: .*frmap.*'):
            _ = DataLayout(origin=PhyDim2(2, 3), frmap=frm,
                           type=NodeRegion.DATA)

    def test_invalid_type(self):
        ''' Invalid type. '''
        with self.assertRaisesRegexp(ValueError, 'DataLayout: .*type.*'):
            _ = DataLayout(origin=PhyDim2(2, 3), frmap=self.frm,
                           type=NodeRegion.NUM)

    def test_is_in_region(self):
        ''' Whether is in region. '''
        nr1 = NodeRegion(dim=PhyDim2(2, 2), origin=PhyDim2(1, 1),
                         type=NodeRegion.DATA)
        self.assertTrue(self.dly.is_in_region(nr1))
        nr2 = NodeRegion(dim=PhyDim2(3, 3), origin=PhyDim2(0, 0),
                         type=NodeRegion.DATA)
        self.assertTrue(self.dly.is_in_region(nr2))
        nr3 = NodeRegion(dim=PhyDim2(2, 2), origin=PhyDim2(0, 0),
                         type=NodeRegion.DATA)
        self.assertFalse(self.dly.is_in_region(nr3))
        nr4 = NodeRegion(dim=PhyDim2(3, 3), origin=PhyDim2(0, 0),
                         type=NodeRegion.PROC)
        self.assertFalse(self.dly.is_in_region(nr4))

        frm = self.frm.copy()
        frm.add(FmapRange((0, 0, 0, 16), (4, 4, 16, 20)),
                (PhyDim2(2, 2), PhyDim2(3, 3)))
        dly = DataLayout(origin=PhyDim2(1, 1), frmap=frm,
                         type=NodeRegion.DATA)

        nr4 = NodeRegion(dim=PhyDim2(3, 3), origin=PhyDim2(1, 1),
                         type=NodeRegion.DATA)
        self.assertFalse(dly.is_in_region(nr4))
        nr5 = NodeRegion(dim=PhyDim2(4, 4), origin=PhyDim2(1, 1),
                         type=NodeRegion.DATA)
        self.assertTrue(dly.is_in_region(nr5))

        frm.add(FmapRange((0, 0, 16, 0), (4, 4, 20, 20)),
                (PhyDim2(1, 1), PhyDim2(3, 3), PhyDim2(5, 5)))
        dly = DataLayout(origin=PhyDim2(1, 1), frmap=frm,
                         type=NodeRegion.DATA)

        self.assertFalse(dly.is_in_region(nr5))
        nr6 = NodeRegion(dim=PhyDim2(7, 7), origin=PhyDim2(0, 0),
                         type=NodeRegion.DATA)
        self.assertTrue(dly.is_in_region(nr6))

    def test_total_transfer_nhops(self):
        ''' Get total_transfer_nhops. '''
        fr = FmapRange((0,) * 4, (4, 4, 16, 16))
        nhops = 2 * 4 * 8 * 16 * (5 + 6 + 6 + 7)
        self.assertEqual(self.dly.total_transfer_nhops(fr, PhyDim2(-1, -2)),
                         nhops, 'total_transfer_nhops')

        frm = self.frm.copy()
        frm.add(FmapRange((0, 0, 0, 16), (4, 4, 16, 20)),
                (PhyDim2(2, 2), PhyDim2(3, 3)))
        frm.add(FmapRange((0, 0, 16, 0), (4, 4, 20, 20)),
                (PhyDim2(1, 1), PhyDim2(3, 3), PhyDim2(5, 5)))
        dly = DataLayout(origin=PhyDim2(1, 1), frmap=frm,
                         type=NodeRegion.DATA)

        self.assertEqual(dly.total_transfer_nhops(fr, PhyDim2(-1, -2)),
                         nhops, 'total_transfer_nhops')

        nhops += 4 * 4 * 16 * 4 * (9 + 11) + 4 * 4 * 4 * 20 * (7 + 11 + 15)
        fr = FmapRange((0,) * 4, (20,) * 4)
        self.assertEqual(dly.total_transfer_nhops(fr, PhyDim2(-1, -2)),
                         nhops, 'total_transfer_nhops')

    def test_view(self):
        ''' Get view. '''
        frm = self.frm.copy()
        frm.add(FmapRange((0, 0, 0, 16), (4, 4, 16, 20)),
                (PhyDim2(2, 2), PhyDim2(3, 3)))
        frm.add(FmapRange((0, 0, 16, 0), (4, 4, 20, 20)),
                (PhyDim2(1, 1), PhyDim2(3, 3), PhyDim2(5, 5)))
        dly = DataLayout(origin=PhyDim2(1, 1), frmap=frm,
                         type=NodeRegion.DATA)

        cfr = dly.frmap.complete_fmap_range()
        counters = dly.frmap.rget_counter(cfr)
        nhops = dly.total_transfer_nhops(cfr, PhyDim2(1, 2))

        dly1 = dly.view(PhyDim2(-1, -1))
        self.assertEqual(dly1.origin, PhyDim2(0, 0), 'view: origin')
        self.assertEqual(dly1.type, dly.type, 'view: type')
        self.assertEqual(dly1.frmap.complete_fmap_range(), cfr,
                         'view: complete_fmap_range')
        self.assertDictEqual(dly1.frmap.rget_counter(cfr), counters,
                             'view: counter')
        self.assertEqual(
            dly1.total_transfer_nhops(cfr, PhyDim2(1, 2) + PhyDim2(-1, -1)),
            nhops, 'view: nhops')

        dly2 = dly.view(PhyDim2(3, 1))
        self.assertEqual(dly2.type, dly.type, 'view: type')
        self.assertEqual(dly2.frmap.complete_fmap_range(), cfr,
                         'view: complete_fmap_range')
        self.assertDictEqual(dly2.frmap.rget_counter(cfr), counters,
                             'view: counter')
        self.assertEqual(
            dly2.total_transfer_nhops(cfr, PhyDim2(1, 2) + PhyDim2(3, 1)),
            nhops, 'view: nhops')

    def test_merge(self):
        ''' Merge. '''
        fr = FmapRange((0,) * 4, (30,) * 4)

        frm = FmapRangeMap()
        frm.add(FmapRange((0, 0, 0, 0), (4, 1, 16, 16)),
                (PhyDim2(0, 0), PhyDim2(1, 1)))
        frm.add(FmapRange((0, 1, 0, 0), (4, 3, 16, 16)),
                (PhyDim2(1, 0), PhyDim2(2, 2)))
        dly = DataLayout(origin=PhyDim2(-1, -1), frmap=frm,
                         type=NodeRegion.DATA)

        mdly1 = self.dly.merge('|', dly)
        mdly2 = dly.merge('|', self.dly)
        self.assertEqual(mdly1.type, self.dly.type, 'merge |: type')
        self.assertEqual(mdly2.type, dly.type, 'merge |: type')
        self.assertEqual(mdly1.frmap.complete_fmap_range(),
                         mdly2.frmap.complete_fmap_range(),
                         'merge |: complete_fmap_range')
        self.assertEqual(mdly1.frmap.complete_fmap_range().size(),
                         self.frm.complete_fmap_range().size()
                         + frm.complete_fmap_range().size(),
                         'merge |: complete_fmap_range: size')
        self.assertEqual(mdly1.total_transfer_nhops(fr, PhyDim2(0, 0)),
                         mdly2.total_transfer_nhops(fr, PhyDim2(0, 0)),
                         'merge |: nhops')
        self.assertEqual(mdly1.total_transfer_nhops(fr, PhyDim2(0, 0)),
                         self.dly.total_transfer_nhops(fr, PhyDim2(0, 0))
                         + dly.total_transfer_nhops(fr, PhyDim2(0, 0)),
                         'merge |: nhops')

        frm.add(FmapRange((0, 3, 0, 0), (4, 4, 16, 16)),
                (PhyDim2(1, 3), PhyDim2(2, 1), PhyDim2(-1, -2)))
        # Only type differs from self.dly.
        sdly = DataLayout(origin=self.dly.origin, frmap=self.dly.frmap,
                          type=NodeRegion.PROC)
        dly = DataLayout(origin=PhyDim2(-1, -1), frmap=frm,
                         type=NodeRegion.PROC)
        mdly1 = sdly.merge('+', dly)
        mdly2 = dly.merge('+', sdly)
        self.assertEqual(mdly1.type, sdly.type, 'merge +: type')
        self.assertEqual(mdly2.type, dly.type, 'merge +: type')
        self.assertEqual(mdly1.frmap.complete_fmap_range(),
                         mdly2.frmap.complete_fmap_range(),
                         'merge +: complete_fmap_range')
        self.assertEqual(mdly1.frmap.complete_fmap_range().size(),
                         self.frm.complete_fmap_range().size(),
                         'merge +: complete_fmap_range: size')
        self.assertEqual(mdly1.total_transfer_nhops(fr, PhyDim2(0, 0)),
                         mdly2.total_transfer_nhops(fr, PhyDim2(0, 0)),
                         'merge +: nhops')
        self.assertEqual(mdly1.total_transfer_nhops(fr, PhyDim2(0, 0)),
                         sdly.total_transfer_nhops(fr, PhyDim2(0, 0))
                         + dly.total_transfer_nhops(fr, PhyDim2(0, 0)),
                         'merge +: nhops')

    def test_merge_invalid_type(self):
        ''' Merge invalid. '''
        with self.assertRaisesRegexp(TypeError, 'DataLayout: .*other.*'):
            _ = self.dly.merge('|', self.frm)
        with self.assertRaisesRegexp(TypeError, 'DataLayout: .*other.*'):
            _ = self.dly.merge('+', PhyDim2(1, 3))

    def test_merge_invalid_region_type(self):
        ''' Merge invalid region type. '''
        dly = DataLayout(origin=self.dly.origin, frmap=self.dly.frmap,
                         type=NodeRegion.PROC)
        with self.assertRaisesRegexp(ValueError, 'DataLayout: .*type.*'):
            _ = self.dly.merge('|', dly)
        with self.assertRaisesRegexp(ValueError, 'DataLayout: .*type.*'):
            _ = self.dly.merge('+', dly)

    def test_merge_invalid_merge_symbol(self):
        ''' Merge invalid merge symbol. '''
        with self.assertRaisesRegexp(ValueError, 'DataLayout: .*symbol.*'):
            _ = self.dly.merge('*', self.dly)
        with self.assertRaisesRegexp(ValueError, 'DataLayout: .*symbol.*'):
            _ = self.dly.merge('^', self.dly)

    def test_merge_unmatch(self):
        ''' Merge unmatch. '''
        for fr in [FmapRange((0,) * 4, (4, 4, 10, 16)),
                   FmapRange((0,) * 4, (4, 4, 16, 32)),
                   FmapRange((0,) * 4, (3, 4, 16, 16))]:
            frm = FmapRangeMap()
            frm.add(fr, (PhyDim2(1, 1),))
            dly = DataLayout(origin=PhyDim2(-1, -1), frmap=frm,
                             type=NodeRegion.DATA)

            with self.assertRaisesRegexp(ValueError, 'DataLayout: .*match.*'):
                _ = self.dly.merge('|', dly)
            with self.assertRaisesRegexp(ValueError, 'DataLayout: .*match.*'):
                _ = self.dly.merge('+', dly)

        frm = FmapRangeMap()
        frm.add(FmapRange((0,) * 4, (4, 2, 16, 16)), (PhyDim2(1, 1),))
        dly = DataLayout(origin=PhyDim2(-1, -1), frmap=frm,
                         type=NodeRegion.DATA)

        with self.assertRaisesRegexp(ValueError, 'DataLayout: .*match.*'):
            _ = self.dly.merge('+', dly)

