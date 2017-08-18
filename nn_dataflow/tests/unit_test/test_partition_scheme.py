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

import collections
import itertools
import unittest

from nn_dataflow.core import Layer, ConvLayer, PoolingLayer
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import PhyDim2

class TestPartitionScheme(unittest.TestCase):
    ''' Tests for PartitionScheme. '''

    def setUp(self):
        self.ps1 = PartitionScheme(order=[pe.BATP, pe.OUTP, pe.OFMP, pe.INPP],
                                   pdims=[(2, 3), (3, 1), (1, 5), (5, 2)])
        self.ps2 = PartitionScheme(order=range(pe.NUM),
                                   pdims=[(2, 2), (5, 5), (3, 3), (1, 1)])

    def test_invalid_order(self):
        ''' Invalid order. '''
        with self.assertRaisesRegexp(ValueError, 'PartitionScheme: .*order.*'):
            _ = PartitionScheme(order=range(pe.NUM - 1),
                                pdims=[(1, 1)] * pe.NUM)

        with self.assertRaisesRegexp(ValueError, 'PartitionScheme: .*order.*'):
            _ = PartitionScheme(order=[0] + range(2, pe.NUM),
                                pdims=[(1, 1)] * pe.NUM)

        with self.assertRaisesRegexp(ValueError, 'PartitionScheme: .*order.*'):
            _ = PartitionScheme(order=[1] + range(pe.NUM),
                                pdims=[(1, 1)] * pe.NUM)

        with self.assertRaisesRegexp(ValueError, 'PartitionScheme: .*order.*'):
            _ = PartitionScheme(order=range(4, 4 + pe.NUM),
                                pdims=[(1, 1)] * pe.NUM)

    def test_invalid_pdims(self):
        ''' Invalid pdims. '''
        with self.assertRaisesRegexp(ValueError, 'PartitionScheme: .*pdims.*'):
            _ = PartitionScheme(order=range(pe.NUM),
                                pdims=[(1, 1)] * (pe.NUM - 1))

        with self.assertRaisesRegexp(ValueError, 'PartitionScheme: .*pdims.*'):
            _ = PartitionScheme(order=range(pe.NUM),
                                pdims=[(1, 1), (1, 1), (2, 1, 1), (1, 1)])

    def test_dim(self):
        ''' Get dim. '''
        self.assertEqual(self.ps1.dim(0), PhyDim2(2, 3))
        self.assertEqual(self.ps1.dim(1), PhyDim2(3, 1))
        self.assertEqual(self.ps1.dim(2), PhyDim2(1, 5))
        self.assertEqual(self.ps1.dim(3), PhyDim2(5, 2))

        self.assertEqual(self.ps2.dim(0), PhyDim2(2, 2))
        self.assertEqual(self.ps2.dim(1), PhyDim2(5, 5))
        self.assertEqual(self.ps2.dim(2), PhyDim2(3, 3))
        self.assertEqual(self.ps2.dim(3), PhyDim2(1, 1))

        self.assertEqual(self.ps1.dim(0, 1, 2),
                         PhyDim2(2, 3) * PhyDim2(3, 1) * PhyDim2(1, 5))
        self.assertEqual(self.ps1.dim(),
                         PhyDim2(2, 3) * PhyDim2(3, 1)
                         * PhyDim2(1, 5) * PhyDim2(5, 2))

        self.assertEqual(self.ps1.dim(0, 1, 2), self.ps1.dim(1, 2, 0))

    def test_dim_invalid_index(self):
        ''' Get dim invalid index. '''
        with self.assertRaises(IndexError):
            _ = self.ps1.dim(pe.NUM + 1)

        with self.assertRaises(IndexError):
            _ = self.ps1.dim(0, 1, pe.NUM)

    def test_size(self):
        ''' Get size. '''
        for l in range(1, pe.NUM):
            for args in itertools.combinations(range(pe.NUM), l):
                self.assertEqual(self.ps1.dim(*args).size(), self.ps1.size(*args))

    def test_size_invalid_index(self):
        ''' Get size invalid index. '''
        with self.assertRaises(IndexError):
            _ = self.ps1.size(pe.NUM + 1)

        with self.assertRaises(IndexError):
            _ = self.ps1.size(0, 1, pe.NUM)

    def test_gen_pidx(self):
        ''' Generate pidx. '''
        for ps in [self.ps1, self.ps2]:

            pidx_list = list(ps.gen_pidx())

            # Num. of pidx == size.
            self.assertEqual(len(pidx_list), ps.size())
            self.assertEqual(len(set(pidx_list)), ps.size())

            for i, idx_list in enumerate(zip(*pidx_list)):
                cnt = collections.Counter(idx_list)
                # Num. of different pidx == size.
                self.assertEqual(len(cnt), ps.size(i))
                # Num. of repeated pidx == other sizes.
                for c in cnt.values():
                    self.assertEqual(c, ps.size() // ps.size(i))

    def test_coordinate(self):
        ''' Get coordinate. '''
        nr1 = NodeRegion(origin=PhyDim2(0, 0), dim=self.ps1.dim(),
                         type=NodeRegion.PROC)
        nr2 = NodeRegion(origin=PhyDim2(0, 0), dim=self.ps2.dim(),
                         type=NodeRegion.PROC)

        for ps, nr in zip([self.ps1, self.ps2], [nr1, nr2]):

            coord_list = [ps.coordinate(nr, pidx) for pidx in ps.gen_pidx()]

            self.assertEqual(len(coord_list), ps.size())
            self.assertEqual(len(set(coord_list)), ps.size())

            for coord in coord_list:
                self.assertGreaterEqual(coord.h, 0)
                self.assertGreaterEqual(coord.w, 0)
                self.assertLess(coord.h, ps.dim().h)
                self.assertLess(coord.w, ps.dim().w)

        pidx = [PhyDim2(0, 0)] * pe.NUM
        pidx[pe.OUTP] = PhyDim2(1, 1)

        self.assertEqual(self.ps1.coordinate(nr1, pidx),
                         self.ps1.dim(pe.OFMP, pe.INPP)
                         * PhyDim2(1, 1))

        self.assertEqual(self.ps2.coordinate(nr2, pidx),
                         self.ps2.dim(pe.OFMP, pe.BATP, pe.INPP)
                         * PhyDim2(1, 1))

    def test_part_layer(self):
        ''' Get part_layer. '''
        batch_size = 16

        layer = ConvLayer(32, 128, 28, 3)
        p_layer, p_batch_size, p_occ = self.ps1.part_layer(layer, batch_size)
        self.assertGreaterEqual(p_layer.hofm * self.ps1.dim(pe.OFMP).h,
                                layer.hofm, 'part_layer: Conv: hofm')
        self.assertGreaterEqual(p_layer.wofm * self.ps1.dim(pe.OFMP).w,
                                layer.wofm, 'part_layer: Conv: wofm')
        self.assertGreaterEqual(p_layer.nofm * self.ps1.size(pe.OUTP),
                                layer.nofm, 'part_layer: Conv: nofm')
        self.assertGreaterEqual(p_layer.nifm * self.ps1.size(pe.INPP),
                                layer.nifm, 'part_layer: Conv: nifm')
        self.assertGreaterEqual(p_batch_size * self.ps1.size(pe.BATP),
                                16, 'part_layer: Conv: batch_size')
        self.assertAlmostEqual(p_occ, 1. * (32 * 128 * 28 * 28 * 16)
                               / (4 * 22 * 10 * 28 * 4 * self.ps1.size()))

        layer = PoolingLayer(128, 112, 2)
        p_layer, p_batch_size, p_occ = self.ps2.part_layer(layer, batch_size)
        self.assertGreaterEqual(p_layer.hofm * self.ps2.dim(pe.OFMP).h,
                                layer.hofm, 'part_layer: Pooling: hofm')
        self.assertGreaterEqual(p_layer.wofm * self.ps2.dim(pe.OFMP).w,
                                layer.wofm, 'part_layer: Pooling: wofm')
        self.assertGreaterEqual(p_layer.nofm * self.ps2.size(pe.OUTP),
                                layer.nofm, 'part_layer: Pooling: nofm')
        self.assertGreaterEqual(p_layer.nifm, p_layer.nofm,
                                'part_layer: Pooling: nifm')
        self.assertGreaterEqual(p_batch_size * self.ps2.size(pe.BATP),
                                16, 'part_layer: Pooling: batch_size')
        self.assertAlmostEqual(p_occ, 1. * (128 * 112 * 112 * 16)
                               / (32 * 23 * 23 * 2 * self.ps2.size()))

    def test_part_layer_invalid_inpart(self):
        ''' Get part_layer invalid INPP. '''
        with self.assertRaisesRegexp(ValueError, 'PartitionScheme: .*input.*'):
            _ = self.ps1.part_layer(PoolingLayer(self.ps1.size(pe.OUTP),
                                                 self.ps1.size(pe.OFMP), 2),
                                    self.ps1.size(pe.BATP))

    def test_part_layer_invalid_type(self):
        ''' Get part_layer invalid type. '''
        class _Layer(Layer):
            def input_layer(self):
                return self
            def ops_per_neuron(self):
                return 0

        with self.assertRaisesRegexp(TypeError, 'PartitionScheme: .*layer.*'):
            _ = self.ps1.part_layer(_Layer(self.ps1.size(pe.OUTP),
                                           self.ps1.size(pe.OFMP)),
                                    self.ps1.size(pe.BATP))

    def test_repr(self):
        ''' __repr__. '''
        # pylint: disable=eval-used
        self.assertEqual(eval(repr(self.ps1)), self.ps1)
        self.assertEqual(eval(repr(self.ps2)), self.ps2)

