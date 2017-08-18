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

from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2

class TestNodeRegion(unittest.TestCase):
    ''' Tests for NodeRegion. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)
        self.assertTupleEqual(nr.dim, (4, 4), 'dim')
        self.assertTupleEqual(nr.origin, (1, 3), 'origin')
        self.assertEqual(nr.type, NodeRegion.PROC, 'type')

    def test_invalid_dim(self):
        ''' Invalid dim. '''
        with self.assertRaisesRegexp(TypeError, 'NodeRegion: .*dim.*'):
            _ = NodeRegion(dim=(4, 4),
                           origin=PhyDim2(1, 3),
                           type=NodeRegion.PROC)

    def test_invalid_origin(self):
        ''' Invalid origin. '''
        with self.assertRaisesRegexp(TypeError, 'NodeRegion: .*origin.*'):
            _ = NodeRegion(dim=PhyDim2(4, 4),
                           origin=(1, 3),
                           type=NodeRegion.PROC)

    def test_invalid_type(self):
        ''' Invalid type. '''
        with self.assertRaisesRegexp(ValueError, 'NodeRegion: .*type.*'):
            _ = NodeRegion(dim=PhyDim2(4, 4),
                           origin=PhyDim2(1, 3),
                           type=NodeRegion.NUM)

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

    def test_node_iter(self):
        ''' Get node iterator. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)
        # No duplicates.
        self.assertEqual(len(set(nr.node_iter())), nr.dim.size())
        # All nodes is contained.
        for c in nr.node_iter():
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
                            set(nr.node_iter()))

    def test_rel2abs_invalid_type(self):
        ''' Get rel2abs invalid type. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)

        with self.assertRaisesRegexp(TypeError, 'NodeRegion: .*PhyDim2.*'):
            _ = nr.rel2abs((0, 0))

        with self.assertRaisesRegexp(TypeError, 'NodeRegion: .*PhyDim2.*'):
            _ = nr.rel2abs(1)

    def test_rel2abs_not_in(self):
        ''' Get rel2abs not in. '''
        nr = NodeRegion(dim=PhyDim2(4, 4),
                        origin=PhyDim2(1, 3),
                        type=NodeRegion.PROC)

        with self.assertRaisesRegexp(ValueError, 'NodeRegion: .*not in.*'):
            _ = nr.rel2abs(PhyDim2(-1, 0))

        with self.assertRaisesRegexp(ValueError, 'NodeRegion: .*not in.*'):
            _ = nr.rel2abs(PhyDim2(0, 4))

