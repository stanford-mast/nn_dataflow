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

import collections
import itertools
import math
import unittest

from nn_dataflow.core import FmapPosition, FmapRange
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
        self.ps2 = PartitionScheme(order=list(range(pe.NUM)),
                                   pdims=[(2, 2), (5, 5), (3, 3), (1, 1)])

        self.nr1 = NodeRegion(origin=PhyDim2(0, 0), dim=self.ps1.dim(),
                              type=NodeRegion.PROC)
        self.nr2 = NodeRegion(origin=PhyDim2(0, 0), dim=self.ps2.dim(),
                              type=NodeRegion.PROC)

    def test_invalid_order(self):
        ''' Invalid order. '''
        with self.assertRaisesRegex(ValueError, 'PartitionScheme: .*order.*'):
            _ = PartitionScheme(order=list(range(pe.NUM - 1)),
                                pdims=[(1, 1)] * pe.NUM)

        with self.assertRaisesRegex(ValueError, 'PartitionScheme: .*order.*'):
            _ = PartitionScheme(order=[0] + list(range(2, pe.NUM)),
                                pdims=[(1, 1)] * pe.NUM)

        with self.assertRaisesRegex(ValueError, 'PartitionScheme: .*order.*'):
            _ = PartitionScheme(order=[1] + list(range(pe.NUM)),
                                pdims=[(1, 1)] * pe.NUM)

        with self.assertRaisesRegex(ValueError, 'PartitionScheme: .*order.*'):
            _ = PartitionScheme(order=list(range(4, 4 + pe.NUM)),
                                pdims=[(1, 1)] * pe.NUM)

    def test_invalid_pdims(self):
        ''' Invalid pdims. '''
        with self.assertRaisesRegex(ValueError, 'PartitionScheme: .*pdims.*'):
            _ = PartitionScheme(order=list(range(pe.NUM)),
                                pdims=[(1, 1)] * (pe.NUM - 1))

        with self.assertRaisesRegex(ValueError, 'PartitionScheme: .*pdims.*'):
            _ = PartitionScheme(order=list(range(pe.NUM)),
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
        for ps, nr in zip([self.ps1, self.ps2], [self.nr1, self.nr2]):

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

        self.assertEqual(self.ps1.coordinate(self.nr1, pidx),
                         self.ps1.dim(pe.OFMP, pe.INPP)
                         * PhyDim2(1, 1))

        self.assertEqual(self.ps2.coordinate(self.nr2, pidx),
                         self.ps2.dim(pe.OFMP, pe.BATP, pe.INPP)
                         * PhyDim2(1, 1))

    def test_fmap_range(self):
        ''' Get fmap_range. '''
        fr1 = FmapRange(FmapPosition(b=0, n=0, h=0, w=0),
                        FmapPosition(b=8, n=64, h=28, w=28))
        # Small ranges.
        fr2 = FmapRange(FmapPosition(b=0, n=0, h=0, w=0),
                        FmapPosition(b=1, n=1, h=1, w=1))
        # Irregular values.
        fr3 = FmapRange(FmapPosition(b=2, n=4, h=2, w=6),
                        FmapPosition(b=5, n=11, h=13, w=13))

        ps = self.ps2

        # No overlap.
        for fr in [fr1, fr2, fr3]:
            pfr_list = [ps.fmap_range(fr, pidx) for pidx in ps.gen_pidx()]
            for idx, pfr in enumerate(pfr_list):
                for jdx in range(idx):
                    self.assertEqual(pfr_list[jdx].overlap_size(pfr), 0)

        pidx = (PhyDim2(1, 0), PhyDim2(4, 3), PhyDim2(0, 2), PhyDim2(0, 0))

        self.assertEqual(ps.fmap_range(fr1, pidx),
                         FmapRange(FmapPosition(b=1, n=32, h=22, w=16),
                                   FmapPosition(b=2, n=48, h=28, w=22)))
        self.assertEqual(ps.fmap_range(fr2, pidx),
                         FmapRange(FmapPosition(b=0, n=0, h=0, w=0),
                                   FmapPosition(b=0, n=0, h=1, w=0)))
        self.assertEqual(ps.fmap_range(fr3, pidx),
                         FmapRange(FmapPosition(b=2, n=7, h=10, w=10),
                                   FmapPosition(b=3, n=9, h=13, w=11)))

    def test_is_appl2frng(self):
        ''' Get is_applicable_to_fmap_range. '''
        self.assertFalse(self.ps1.is_applicable_to_fmap_range())
        self.assertTrue(self.ps2.is_applicable_to_fmap_range())

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
        with self.assertRaisesRegex(ValueError, 'PartitionScheme: .*input.*'):
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
            @staticmethod
            def data_loops():
                return None

        layer = _Layer(self.ps1.size(pe.OUTP), self.ps1.size(pe.OFMP))
        self.assertEqual(layer.total_ops(), 0)
        self.assertIsNone(_Layer.data_loops())

        with self.assertRaisesRegex(TypeError, 'PartitionScheme: .*layer.*'):
            _ = self.ps1.part_layer(layer, self.ps1.size(pe.BATP))

    def test_part_neighbor_dist(self):
        ''' Get part_neighbor_dist. '''
        for ps, nr in zip([self.ps1, self.ps2], [self.nr1, self.nr2]):

            for idx in range(pe.NUM):
                nbr_dist = ps.part_neighbor_dist(nr, ps.order[idx])
                dim_below = ps.dim(*ps.order[idx + 1:]) if idx + 1 < pe.NUM \
                        else PhyDim2(1, 1)
                dim_cur = ps.dim(ps.order[idx])

                if dim_cur.h == 1:
                    self.assertTrue(math.isinf(nbr_dist.h))
                else:
                    self.assertEqual(nbr_dist.h, dim_below.h)

                if dim_cur.w == 1:
                    self.assertTrue(math.isinf(nbr_dist.w))
                else:
                    self.assertEqual(nbr_dist.w, dim_below.w)

    def test_part_neighbor_dist_inv(self):
        ''' Get part_neighbor_dist invalid arg. '''
        dist = self.ps1.part_neighbor_dist(self.nr1, pe.NUM)
        self.assertTrue(all(math.isnan(d) for d in dist))

    def test_projection(self):
        ''' Get projection. '''

        def _make_region(dim):
            return NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(*dim),
                              type=NodeRegion.DRAM)

        # Shrink.
        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (4, 4), (1, 1)))
        proj_part = part.projection(_make_region((4, 30)))
        self.assertTupleEqual(proj_part.order, part.order)
        self.assertTupleEqual(proj_part.dim(), (4, 30))
        self.assertTupleEqual(proj_part.dim(pe.OUTP), (2, 3))
        self.assertTupleEqual(proj_part.dim(pe.OFMP), (1, 5))
        self.assertTupleEqual(proj_part.dim(pe.BATP), (2, 2))
        self.assertTupleEqual(proj_part.dim(pe.INPP), (1, 1))

        # Shrink multiple.
        proj_part = part.projection(_make_region((2, 2)))
        self.assertTupleEqual(proj_part.order, part.order)
        self.assertTupleEqual(proj_part.dim(), (2, 2))
        self.assertTupleEqual(proj_part.dim(pe.OUTP), (2, 1))
        self.assertTupleEqual(proj_part.dim(pe.OFMP), (1, 2))
        self.assertTupleEqual(proj_part.dim(pe.BATP), (1, 1))
        self.assertTupleEqual(proj_part.dim(pe.INPP), (1, 1))

        # Shrink non-dividable.
        proj_part = part.projection(_make_region((3, 54)))
        self.assertTupleEqual(proj_part.order, part.order)
        self.assertTupleEqual(proj_part.dim(), (2, 45))
        self.assertTupleEqual(proj_part.dim(pe.OUTP), (2, 3))
        self.assertTupleEqual(proj_part.dim(pe.OFMP), (1, 5))
        # For height, 3 // 2 = 1.
        # For width, 54 // 5 = 10, 10 // 3 = 3.
        self.assertTupleEqual(proj_part.dim(pe.BATP), (1, 3))
        self.assertTupleEqual(proj_part.dim(pe.INPP), (1, 1))

        # Shrink with INPP.
        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (4, 4), (4, 4)))
        proj_part = part.projection(_make_region((4, 30)), appl2frng=True)
        self.assertTupleEqual(proj_part.order, part.order)
        self.assertTupleEqual(proj_part.dim(), (4, 30))
        self.assertTupleEqual(proj_part.dim(pe.BATP), (2, 2))
        self.assertTupleEqual(proj_part.dim(pe.INPP), (1, 1))
        proj_part = part.projection(_make_region((4, 30)))
        self.assertTupleEqual(proj_part.order, part.order)
        self.assertTupleEqual(proj_part.dim(), (4, 30))
        self.assertTupleEqual(proj_part.dim(pe.BATP), (1, 1))
        self.assertTupleEqual(proj_part.dim(pe.INPP), (2, 2))

        # Shrink all.
        proj_part = part.projection(_make_region((1, 1)))
        self.assertTupleEqual(proj_part.order, part.order)
        self.assertTupleEqual(proj_part.dim(), (1, 1))

        # Extend.
        part = PartitionScheme(order=(pe.INPP, pe.BATP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (1, 1), (1, 1)))
        proj_part = part.projection(_make_region((4, 30)))
        self.assertTupleEqual(proj_part.order, part.order)
        self.assertTupleEqual(proj_part.dim(), (4, 30))
        self.assertTupleEqual(proj_part.dim(pe.OUTP), (2, 3))
        self.assertTupleEqual(proj_part.dim(pe.OFMP), (1, 5))
        self.assertTupleEqual(proj_part.dim(pe.BATP), (2, 2))
        self.assertTupleEqual(proj_part.dim(pe.INPP), (1, 1))

        # Extend non-dividable.
        proj_part = part.projection(_make_region((5, 40)))
        self.assertTupleEqual(proj_part.order, part.order)
        self.assertTupleEqual(proj_part.dim(), (4, 30))
        self.assertTupleEqual(proj_part.dim(pe.OUTP), (2, 3))
        self.assertTupleEqual(proj_part.dim(pe.OFMP), (1, 5))
        # For height, 5 // 2 = 2.
        # For width, 40 // (3 * 5) == 2.
        self.assertTupleEqual(proj_part.dim(pe.BATP), (2, 2))
        self.assertTupleEqual(proj_part.dim(pe.INPP), (1, 1))

        # Extend with INPP.
        part = PartitionScheme(order=(pe.INPP, pe.BATP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (1, 1), (4, 4)))
        proj_part = part.projection(_make_region((4, 30)), appl2frng=True)
        self.assertTupleEqual(proj_part.order, part.order)
        self.assertTupleEqual(proj_part.dim(), (4, 30))
        self.assertTupleEqual(proj_part.dim(pe.OUTP), (2, 3))
        self.assertTupleEqual(proj_part.dim(pe.OFMP), (1, 5))
        self.assertTupleEqual(proj_part.dim(pe.BATP), (2, 2))
        self.assertTupleEqual(proj_part.dim(pe.INPP), (1, 1))

        # Both shrink and extend.
        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (4, 4), (1, 1)))
        proj_part = part.projection(_make_region((16, 16)))
        self.assertTupleEqual(proj_part.order, part.order)
        self.assertTupleEqual(proj_part.dim(), (16, 15))
        self.assertTupleEqual(proj_part.dim(pe.OUTP), (2, 3))
        self.assertTupleEqual(proj_part.dim(pe.OFMP), (1, 5))
        self.assertTupleEqual(proj_part.dim(pe.BATP), (8, 1))
        self.assertTupleEqual(proj_part.dim(pe.INPP), (1, 1))

    def test_projection_empty_region(self):
        ''' Get projection with empty region. '''
        with self.assertRaisesRegex(ValueError, 'PartitionScheme: .*region.*'):
            _ = self.ps1.projection(NodeRegion(origin=PhyDim2(0, 0),
                                               dim=PhyDim2(0, 0),
                                               type=NodeRegion.DRAM))

    def test_repr(self):
        ''' __repr__. '''
        # pylint: disable=eval-used
        self.assertEqual(eval(repr(self.ps1)), self.ps1)
        self.assertEqual(eval(repr(self.ps2)), self.ps2)

