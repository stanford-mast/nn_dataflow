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

from nn_dataflow.core import PhyDim2

class TestPhyDim2(unittest.TestCase):
    ''' Tests for PhyDim2. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        dim = PhyDim2(14, 12)
        self.assertEqual(dim.h, 14, 'h')
        self.assertEqual(dim.w, 12, 'w')

    def test_size(self):
        ''' Get size. '''
        dim = PhyDim2(14, 12)
        self.assertEqual(dim.size(), 14 * 12, 'size')

    def test_add(self):
        ''' Operation add. '''
        dim1 = PhyDim2(14, 12)
        dim2 = PhyDim2(5, 3)
        self.assertTupleEqual(dim1 + dim2, (19, 15), 'add')
        self.assertTupleEqual(dim1 + 3, (17, 15), 'add')

    def test_sub(self):
        ''' Operation sub. '''
        dim1 = PhyDim2(14, 12)
        dim2 = PhyDim2(5, 3)
        self.assertTupleEqual(dim1 - dim2, (9, 9), 'sub')
        self.assertTupleEqual(dim1 - 3, (11, 9), 'sub')

    def test_neg(self):
        ''' Operation neg. '''
        dim1 = PhyDim2(14, 12)
        dim2 = PhyDim2(5, 3)
        self.assertTupleEqual(-dim1, (-14, -12), 'neg')
        self.assertTupleEqual(-dim2, (-5, -3), 'neg')

    def test_mul(self):
        ''' Operation mul. '''
        dim1 = PhyDim2(14, 12)
        dim2 = PhyDim2(5, 3)
        self.assertTupleEqual(dim1 * dim2, (70, 36), 'mul')
        self.assertTupleEqual(dim1 * 2, (28, 24), 'mul')
        self.assertTupleEqual(2 * dim1, (28, 24), 'rmul')

    def test_hop_dist(self):
        ''' Get hop distance. '''
        dim1 = PhyDim2(14, 12)
        dim2 = PhyDim2(5, 20)
        self.assertEqual(dim1.hop_dist(dim2), 9 + 8, 'hop_dist')
        self.assertEqual(dim2.hop_dist(dim1), 9 + 8, 'hop_dist')

    def test_hop_dist_error(self):
        ''' Get hop distance. '''
        dim1 = PhyDim2(14, 12)
        with self.assertRaisesRegex(TypeError, 'hop_dist'):
            _ = dim1.hop_dist((5, 20))

