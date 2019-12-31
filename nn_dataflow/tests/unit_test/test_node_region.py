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

from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2

class TestNodeRegion(unittest.TestCase):
    ''' Tests for NodeRegion. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC,
                        wtot=2,
                        wbeg=-1)
        self.assertTupleEqual(nr.dim, (4, 4), 'dim')
        self.assertTupleEqual(nr.origin, (1, 3), 'origin')
        self.assertEqual(nr.type, NodeRegion.PROC, 'type')
        self.assertEqual(nr.wtot, 2, 'wtot')
        self.assertEqual(nr.wbeg, -1, 'wbeg')

    def test_default_wtot_wbeg(self):
        ''' Default wtot and wbeg. '''
        nr = NodeRegion(dim=PhyDim2(4, 8),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)
        self.assertEqual(nr.wtot, 8)
        self.assertEqual(nr.wbeg, 8)

        nr = NodeRegion(dim=PhyDim2(4, 8),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC,
                        wtot=6)
        self.assertEqual(nr.wtot, 6)
        self.assertEqual(nr.wbeg, 6)

        nr = NodeRegion(dim=PhyDim2(4, 8),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC,
                        wbeg=-5)
        self.assertEqual(nr.wtot, 8)
        self.assertEqual(nr.wbeg, -5)

    def test_args_kwargs(self):
        ''' Different ways to give args and kwargs. '''
        dim = PhyDim2(4, 8)
        origin = PhyDim2(1, 3)
        dist = PhyDim2(1, 1)
        type_ = NodeRegion.PROC
        wtot = 6
        wbeg = 5

        nr0 = NodeRegion(dim=dim, origin=origin, dist=dist, type=type_,
                         wtot=wtot, wbeg=wbeg)

        nr = NodeRegion(dim, origin, dist, type_, wtot, wbeg)
        self.assertTupleEqual(nr, nr0)

        nr = NodeRegion(dim, origin, wbeg=wbeg, wtot=wtot, type=type_,
                        dist=dist)
        self.assertTupleEqual(nr, nr0)

        nr = NodeRegion(dim, origin, dist, type=type_, wtot=wtot, wbeg=wbeg)
        self.assertTupleEqual(nr, nr0)

    def test_larger_wtot(self):
        ''' wtot > dim.w is valid. '''
        nr = NodeRegion(dim=PhyDim2(4, 8),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC,
                        wtot=20)
        self.assertEqual(nr.wtot, 20)

    def test_invalid_dim(self):
        ''' Invalid dim. '''
        with self.assertRaisesRegex(TypeError, 'NodeRegion: .*dim.*'):
            _ = NodeRegion(dim=(4, 4),
                           origin=PhyDim2(1, 3),
                           type=NodeRegion.PROC)

    def test_invalid_origin(self):
        ''' Invalid origin. '''
        with self.assertRaisesRegex(TypeError, 'NodeRegion: .*origin.*'):
            _ = NodeRegion(dim=PhyDim2(4, 4),
                           origin=(1, 3),
                           type=NodeRegion.PROC)

    def test_invalid_dist(self):
        ''' Invalid dist. '''
        with self.assertRaisesRegex(TypeError, 'NodeRegion: .*dist.*'):
            _ = NodeRegion(dim=PhyDim2(4, 4),
                           origin=PhyDim2(1, 3),
                           dist=(1, 1),
                           type=NodeRegion.PROC)

    def test_invalid_type(self):
        ''' Invalid type. '''
        with self.assertRaisesRegex(ValueError, 'NodeRegion: .*type.*'):
            _ = NodeRegion(dim=PhyDim2(4, 4),
                           origin=PhyDim2(1, 3),
                           type=NodeRegion.NUM)

    def test_invalid_wtot_type(self):
        ''' Invalid wtot type. '''
        with self.assertRaisesRegex(TypeError, 'NodeRegion: .*wtot.*'):
            _ = NodeRegion(dim=PhyDim2(4, 4),
                           origin=PhyDim2(1, 3),
                           type=NodeRegion.PROC,
                           wtot=1.3)

    def test_invalid_wbeg_type(self):
        ''' Invalid wbeg type. '''
        with self.assertRaisesRegex(TypeError, 'NodeRegion: .*wbeg.*'):
            _ = NodeRegion(dim=PhyDim2(4, 4),
                           origin=PhyDim2(1, 3),
                           type=NodeRegion.PROC,
                           wbeg=1.3)

    def test_invalid_wbeg(self):
        ''' Invalid wbeg. '''
        with self.assertRaisesRegex(ValueError, 'NodeRegion: .*wbeg.*'):
            _ = NodeRegion(dim=PhyDim2(4, 4),
                           origin=PhyDim2(1, 3),
                           type=NodeRegion.PROC,
                           wtot=4,
                           wbeg=5)

        with self.assertRaisesRegex(ValueError, 'NodeRegion: .*wbeg.*'):
            _ = NodeRegion(dim=PhyDim2(4, 4),
                           origin=PhyDim2(1, 3),
                           type=NodeRegion.PROC,
                           wtot=4,
                           wbeg=-5)

        with self.assertRaisesRegex(ValueError, 'NodeRegion: .*wbeg.*'):
            _ = NodeRegion(dim=PhyDim2(4, 4),
                           origin=PhyDim2(1, 3),
                           type=NodeRegion.PROC,
                           wtot=4,
                           wbeg=0)

    def test_contains_node(self):
        ''' Whether contains node. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)
        for h in range(1, 5):
            for w in range(3, 7):
                self.assertTrue(nr.contains_node(PhyDim2(h, w)))

        num = 0
        for h in range(-2, 10):
            for w in range(-2, 10):
                num += 1 if nr.contains_node(PhyDim2(h, w)) else 0
        self.assertEqual(num, nr.dim.size())

    def test_iter_node(self):
        ''' Get node iterator. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)
        # No duplicates.
        self.assertEqual(len(set(nr.iter_node())), nr.dim.size())
        # All nodes is contained.
        for c in nr.iter_node():
            self.assertTrue(nr.contains_node(c))

    def test_rel2abs(self):
        ''' Get rel2abs. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)

        self.assertTupleEqual(nr.rel2abs(PhyDim2(0, 3)), (1, 6))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(2, 1)), (3, 4))

        self.assertSetEqual(set(nr.rel2abs(PhyDim2(h, w))
                                for h in range(nr.dim.h)
                                for w in range(nr.dim.w)),
                            set(nr.iter_node()))

    def test_rel2abs_dist(self):
        ''' Get rel2abs dist. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        dist=PhyDim2(5, 3),
                        type=NodeRegion.PROC)

        self.assertTupleEqual(nr.rel2abs(PhyDim2(0, 3)), (1, 12))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(2, 1)), (11, 6))

        self.assertSetEqual(set(nr.rel2abs(PhyDim2(h, w))
                                for h in range(nr.dim.h)
                                for w in range(nr.dim.w)),
                            set(nr.iter_node()))

    def test_rel2abs_invalid_type(self):
        ''' Get rel2abs invalid type. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)

        with self.assertRaisesRegex(TypeError, 'NodeRegion: .*PhyDim2.*'):
            _ = nr.rel2abs((0, 0))

        with self.assertRaisesRegex(TypeError, 'NodeRegion: .*PhyDim2.*'):
            _ = nr.rel2abs(1)

    def test_rel2abs_not_in(self):
        ''' Get rel2abs not in. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)

        with self.assertRaisesRegex(ValueError, 'NodeRegion: .*not in.*'):
            _ = nr.rel2abs(PhyDim2(-1, 0))

        with self.assertRaisesRegex(ValueError, 'NodeRegion: .*not in.*'):
            _ = nr.rel2abs(PhyDim2(0, 4))

    def test_rel2abs_folded(self):
        ''' Get rel2abs with folded. '''
        nr = NodeRegion(dim=PhyDim2(4, 8),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC,
                        wtot=3)
        # 67
        # 543
        # 012

        self.assertTupleEqual(nr.rel2abs(PhyDim2(1, 2)), (1 + 1, 5))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(2, 3)), (5 + 2, 5))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(0, 5)), (5 + 0, 3))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(3, 7)), (9 + 3, 4))

        self.assertSetEqual(set(nr.rel2abs(PhyDim2(h, w))
                                for h in range(nr.dim.h)
                                for w in range(nr.dim.w)),
                            set(nr.iter_node()))

        nr = NodeRegion(dim=PhyDim2(4, 8),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC,
                        wtot=3,
                        wbeg=1)
        #   7
        # 456
        # 321
        #   0

        self.assertTupleEqual(nr.rel2abs(PhyDim2(2, 0)), (1 + 2, 3))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(1, 2)), (5 + 1, 2))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(2, 3)), (5 + 2, 1))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(0, 5)), (9 + 0, 2))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(3, 7)), (13 + 3, 3))

        self.assertSetEqual(set(nr.rel2abs(PhyDim2(h, w))
                                for h in range(nr.dim.h)
                                for w in range(nr.dim.w)),
                            set(nr.iter_node()))

        nr = NodeRegion(dim=PhyDim2(4, 8),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC,
                        wtot=4,
                        wbeg=-2)
        #   76
        # 2345
        # 10

        self.assertTupleEqual(nr.rel2abs(PhyDim2(1, 1)), (1 + 1, 2))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(2, 3)), (5 + 2, 3))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(0, 5)), (5 + 0, 5))
        self.assertTupleEqual(nr.rel2abs(PhyDim2(3, 7)), (9 + 3, 4))

        self.assertSetEqual(set(nr.rel2abs(PhyDim2(h, w))
                                for h in range(nr.dim.h)
                                for w in range(nr.dim.w)),
                            set(nr.iter_node()))

    def test_allocate(self):
        ''' allocate. '''

        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)

        def _common_check(length):
            self.assertEqual(len(subregions), length)
            aggr_node_set = set()
            for sr in subregions:
                self.assertTupleEqual(sr.dist, nr.dist)
                self.assertEqual(sr.type, NodeRegion.PROC)
                self.assertEqual(sr.wtot, 4)
                for c in sr.iter_node():
                    self.assertTrue(nr.contains_node(c))
                self.assertTrue(aggr_node_set.isdisjoint(sr.iter_node()))
                aggr_node_set.update(sr.iter_node())
            self.assertSetEqual(set(nr.iter_node()), aggr_node_set)

        request_list = [4, 4, 4, 4, 4]
        self.assertEqual(len(nr.allocate(request_list)), 0)

        request_list = [2, 3, 3, 2, 4, 2]
        subregions = nr.allocate(request_list)
        # 5544
        # 3344
        # 2221
        # 0011
        _common_check(len(request_list))
        self.assertTupleEqual(subregions[0].dim, (1, 2))
        self.assertTupleEqual(subregions[0].origin, (1, 3))
        self.assertEqual(subregions[0].wbeg, 2)
        self.assertTupleEqual(subregions[1].dim, (1, 3))
        self.assertTupleEqual(subregions[1].origin, (1, 5))
        self.assertEqual(subregions[1].wbeg, 2)
        self.assertTupleEqual(subregions[2].dim, (1, 3))
        self.assertTupleEqual(subregions[2].origin, (2, 5))
        self.assertEqual(subregions[2].wbeg, -3)
        self.assertTupleEqual(subregions[3].dim, (1, 2))
        self.assertTupleEqual(subregions[3].origin, (3, 3))
        self.assertEqual(subregions[3].wbeg, 2)
        self.assertTupleEqual(subregions[4].dim, (1, 4))
        self.assertTupleEqual(subregions[4].origin, (3, 5))
        self.assertEqual(subregions[4].wbeg, 2)
        self.assertTupleEqual(subregions[5].dim, (1, 2))
        self.assertTupleEqual(subregions[5].origin, (4, 4))
        self.assertEqual(subregions[5].wbeg, -2)

        request_list = [5, 11]
        subregions = nr.allocate(request_list)
        # 1111
        # 1111
        # 1110
        # 0000
        _common_check(len(request_list))
        self.assertTupleEqual(subregions[0].dim, (1, 5))
        self.assertTupleEqual(subregions[0].origin, (1, 3))
        self.assertEqual(subregions[0].wbeg, 4)
        self.assertTupleEqual(subregions[1].dim, (1, 11))
        self.assertTupleEqual(subregions[1].origin, (2, 5))
        self.assertEqual(subregions[1].wbeg, -3)

        request_list = [2, 4, 4, 2, 4]
        subregions = nr.allocate(request_list)
        # 4432
        # 4432
        # 0112
        # 0112
        _common_check(len(request_list))
        self.assertTupleEqual(subregions[0].dim, (2, 1))
        self.assertTupleEqual(subregions[0].origin, (1, 3))
        self.assertEqual(subregions[0].wbeg, 1)
        self.assertTupleEqual(subregions[1].dim, (2, 2))
        self.assertTupleEqual(subregions[1].origin, (1, 4))
        self.assertEqual(subregions[1].wbeg, 2)
        self.assertTupleEqual(subregions[2].dim, (2, 2))
        self.assertTupleEqual(subregions[2].origin, (1, 6))
        self.assertEqual(subregions[2].wbeg, 1)
        self.assertTupleEqual(subregions[3].dim, (2, 1))
        self.assertTupleEqual(subregions[3].origin, (3, 5))
        self.assertEqual(subregions[3].wbeg, -1)
        self.assertTupleEqual(subregions[4].dim, (2, 2))
        self.assertTupleEqual(subregions[4].origin, (3, 4))
        self.assertEqual(subregions[4].wbeg, -2)

        nr = nr._replace(dist=PhyDim2(2, 1))

        request_list = [10, 6]
        subregions = nr.allocate(request_list)
        # 1110
        # 1110
        # 0000
        # 0000
        _common_check(len(request_list))
        self.assertTupleEqual(subregions[0].dim, (2, 5))
        self.assertTupleEqual(subregions[0].origin, (1, 3))
        self.assertEqual(subregions[0].wbeg, 4)
        self.assertTupleEqual(subregions[1].dim, (2, 3))
        self.assertTupleEqual(subregions[1].origin, (5, 5))
        self.assertEqual(subregions[1].wbeg, -3)

