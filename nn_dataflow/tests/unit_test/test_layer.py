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

from nn_dataflow.core import Layer
from nn_dataflow.core import InputLayer
from nn_dataflow.core import ConvLayer, FCLayer
from nn_dataflow.core import LocalRegionLayer, PoolingLayer, EltwiseLayer
from nn_dataflow.core import DataCategoryEnum as de
from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import DataDimLoops

class TestLayer(unittest.TestCase):
    ''' Tests for Layer. '''

    def test_valid_args(self):
        ''' Valid argument. '''
        clayer = ConvLayer(3, 64, 28, 3, strd=2)
        self.assertEqual(clayer.nifm, 3, 'nifm')
        self.assertEqual(clayer.nofm, 64, 'nofm')
        self.assertEqual(clayer.hofm, 28, 'hofm')
        self.assertEqual(clayer.wofm, 28, 'wofm')
        self.assertEqual(clayer.hfil, 3, 'hfil')
        self.assertEqual(clayer.wfil, 3, 'wfil')
        self.assertEqual(clayer.htrd, 2, 'htrd')
        self.assertEqual(clayer.wtrd, 2, 'wtrd')

        llayer = LocalRegionLayer(64, 28, 1, 3, strd=2)
        self.assertEqual(llayer.nofm, 64, 'nofm')
        self.assertEqual(llayer.hofm, 28, 'hofm')
        self.assertEqual(llayer.wofm, 28, 'wofm')
        self.assertEqual(llayer.nreg, 1, 'nreg')
        self.assertEqual(llayer.hreg, 3, 'hreg')
        self.assertEqual(llayer.wreg, 3, 'wreg')
        self.assertEqual(llayer.htrd, 2, 'htrd')
        self.assertEqual(llayer.wtrd, 2, 'wtrd')

    def test_diff_hwofm(self):
        ''' Different h/w for ofm. '''
        clayer = ConvLayer(3, 64, [28, 14], 3)
        self.assertEqual(clayer.hofm, 28, 'hofm')
        self.assertEqual(clayer.wofm, 14, 'wofm')

    def test_invalid_sofm(self):
        ''' Invalid sofm. '''
        with self.assertRaisesRegex(ValueError, 'Layer: .*sofm.*'):
            _ = ConvLayer(3, 64, [28, 14, 7], 3)

    def test_diff_hwtrd(self):
        ''' Different h/w for stride. '''
        clayer = ConvLayer(3, 64, 28, 3, strd=[2, 4])
        self.assertEqual(clayer.htrd, 2, 'htrd')
        self.assertEqual(clayer.wtrd, 4, 'wtrd')

    def test_invalid_strd(self):
        ''' Invalid stride. '''
        with self.assertRaisesRegex(ValueError, 'Layer: .*strd.*'):
            _ = ConvLayer(3, 64, 28, 3, strd=[2, 3, 4])

    def test_ifmap(self):
        ''' Get ifmap. '''
        clayer = ConvLayer(3, 64, [28, 14], 3, strd=2)
        inlayer = clayer.input_layer()
        self.assertIsInstance(inlayer, Layer)
        self.assertEqual(inlayer.nofm, clayer.nifm, 'ConvLayer: nifm')
        self.assertEqual(inlayer.hofm, clayer.hifm, 'ConvLayer: hifm')
        self.assertEqual(inlayer.wofm, clayer.wifm, 'ConvLayer: wifm')

        llayer = LocalRegionLayer(64, 28, 1, 3, strd=2)
        inlayer = llayer.input_layer()
        self.assertIsInstance(inlayer, Layer)
        self.assertEqual(inlayer.nofm, llayer.nifm, 'LocalRegionLayer: nifm')
        self.assertEqual(inlayer.hofm, llayer.hifm, 'LocalRegionLayer: hifm')
        self.assertEqual(inlayer.wofm, llayer.wifm, 'LocalRegionLayer: wifm')

    def test_ofmap_size(self):
        ''' Get ofmap size. '''
        clayer = ConvLayer(3, 64, [28, 14], 3)
        self.assertEqual(clayer.ofmap_size(), 28 * 14, 'ofmap_size')
        self.assertEqual(clayer.ofmap_size(2), 28 * 14 * 2, 'ofmap_size')
        self.assertEqual(clayer.total_ofmap_size(), 28 * 14 * 64,
                         'total_ofmap_size')
        self.assertEqual(clayer.total_ofmap_size(2), 28 * 14 * 64 * 2,
                         'total_ofmap_size')

    def test_ifmap_size(self):
        ''' Get ifmap size. '''
        clayer = ConvLayer(3, 64, [28, 14], 3, strd=2)
        self.assertEqual(clayer.ifmap_size(2),
                         ((28 - 1) * 2 + 3) * ((14 - 1) * 2 + 3) * 2,
                         'ConvLayer: ifmap_size')
        self.assertEqual(clayer.total_ifmap_size(2),
                         ((28 - 1) * 2 + 3) * ((14 - 1) * 2 + 3) * 3 * 2,
                         'ConvLayer: total_ifmap_size')

        llayer = LocalRegionLayer(64, 28, 1, 3, strd=2)
        self.assertEqual(llayer.ifmap_size(2),
                         ((28 - 1) * 2 + 3) ** 2 * 2,
                         'LocalRegionLayer: ifmap_size')
        self.assertEqual(llayer.total_ifmap_size(2),
                         ((28 - 1) * 2 + 3) ** 2 * 64 * 2,
                         'LocalRegionLayer: total_ifmap_size')

    def test_data_loops(self):
        ''' Get data_loops. '''
        with self.assertRaises(NotImplementedError):
            _ = Layer.data_loops()
        layer = Layer(64, 28)
        with self.assertRaises(NotImplementedError):
            _ = layer.data_loops()

    def test_input_layer(self):
        ''' Get input layer. '''
        layer = Layer(64, 28)
        with self.assertRaises(NotImplementedError):
            _ = layer.input_layer()

    def test_ops(self):
        ''' Get ops. '''
        layer = Layer(64, 28)
        with self.assertRaises(NotImplementedError):
            _ = layer.ops_per_neuron()

    def test_is_valid_padding_sifm(self):
        ''' is_valid_padding_sifm. '''
        clayer = ConvLayer(3, 64, [28, 14], [3, 1], [2, 4])
        self.assertTrue(clayer.is_valid_padding_sifm([28 * 2, 14 * 4]))
        self.assertTrue(clayer.is_valid_padding_sifm([27 * 2 + 3, 13 * 4 + 1]))
        self.assertFalse(clayer.is_valid_padding_sifm([28, 14]))
        self.assertFalse(clayer.is_valid_padding_sifm([28 * 2, 14]))
        self.assertTrue(clayer.is_valid_padding_sifm([27 * 2 + 3, 13 * 4 + 3]))

        flayer = FCLayer(2048, 4096, sfil=2)
        self.assertTrue(flayer.is_valid_padding_sifm(2))
        self.assertTrue(flayer.is_valid_padding_sifm(1))
        self.assertTrue(flayer.is_valid_padding_sifm([1, 2]))

        llayer = LocalRegionLayer(64, 28, 2, 1)
        self.assertTrue(llayer.is_valid_padding_sifm(28))
        self.assertFalse(llayer.is_valid_padding_sifm(28 - 1))
        self.assertFalse(llayer.is_valid_padding_sifm(28 + 1))

        player = PoolingLayer(64, 28, [2, 3], strd=[3, 2])
        self.assertTrue(player.is_valid_padding_sifm([28 * 3, 28 * 2]))
        self.assertTrue(player.is_valid_padding_sifm([27 * 3 + 2, 27 * 2 + 3]))

    def test_is_valid_padding_sifm_inv(self):
        ''' Invalid argument for is_valid_padding_sifm. '''
        clayer = ConvLayer(3, 64, 28, 3, strd=2)
        with self.assertRaisesRegex(ValueError, 'Layer: .*sifm.*'):
            _ = clayer.is_valid_padding_sifm([3])

    def test_eq(self):
        ''' Whether eq. '''
        l1 = Layer(2, 12)
        l2 = Layer(2, 12)
        self.assertEqual(l1, l2)

        l1 = ConvLayer(2, 12, 56, 3)
        l2 = ConvLayer(2, 12, 56, 3)
        self.assertEqual(l1, l2)

        l1 = PoolingLayer(12, 14, 2)
        l2 = PoolingLayer(12, 14, 2)
        self.assertEqual(l1, l2)

        _ = l1 == 4

    def test_ne(self):
        ''' Whether ne. '''
        l1 = Layer(4, 12)
        l2 = Layer(4, 13)
        self.assertNotEqual(l1, l2)
        self.assertNotEqual(l1, (4, 12))

    def test_hash(self):
        ''' Get hash. '''
        l1 = Layer(2, 12)
        l2 = Layer(2, 12)
        self.assertEqual(hash(l1), hash(l2))

        l1 = ConvLayer(2, 12, 56, 3)
        l2 = ConvLayer(2, 12, 56, 3)
        self.assertEqual(hash(l1), hash(l2))

        l1 = PoolingLayer(12, 14, 2)
        l2 = PoolingLayer(12, 14, 2)
        self.assertEqual(hash(l1), hash(l2))

    def test_repr(self):
        ''' __repr__. '''
        # pylint: disable=eval-used
        for l in [Layer(4, 12), Layer(4, [12, 24]), Layer(4, 12, strd=3),
                  Layer(4, 12, strd=[3, 1]), Layer(4, [12, 24], strd=[3, 1])]:
            self.assertIn('Layer', repr(l))
            self.assertEqual(eval(repr(l)), l)


class TestInputLayer(unittest.TestCase):
    ''' Tests for InputLayer. '''

    def test_data_loops(self):
        ''' Get data_loops. '''
        dls = InputLayer.data_loops()
        ilayer = InputLayer(3, 227)
        self.assertTupleEqual(ilayer.data_loops(), dls)
        self.assertEqual(dls[de.FIL], DataDimLoops())
        self.assertEqual(dls[de.IFM], DataDimLoops())
        self.assertEqual(dls[de.OFM], DataDimLoops(le.OFM, le.BAT))

    def test_input_layer(self):
        ''' Get input layer. '''
        ilayer = InputLayer(3, 227)
        self.assertIsNone(ilayer.input_layer(), 'InputLayer: input_layer')

    def test_ops(self):
        ''' Get ops. '''
        ilayer = InputLayer(3, 227)
        self.assertEqual(ilayer.ops_per_neuron(), 0,
                         'InputLayer: ops_per_neurons')
        self.assertEqual(ilayer.total_ops(), 0, 'InputLayer: total_ops')

    def test_repr(self):
        ''' __repr__. '''
        # pylint: disable=eval-used
        for l in [InputLayer(4, 12), InputLayer(4, [12, 24]),
                  InputLayer(4, 12, strd=3), InputLayer(4, 12, strd=[3, 1]),
                  InputLayer(4, [12, 24], strd=[3, 1])]:
            self.assertIn('InputLayer', repr(l))
            self.assertEqual(eval(repr(l)), l)


class TestConvLayer(unittest.TestCase):
    ''' Tests for ConvLayer. '''

    def test_data_loops(self):
        ''' Get data_loops. '''
        dls = ConvLayer.data_loops()
        self.assertEqual(dls[de.FIL], DataDimLoops(le.IFM, le.OFM))
        self.assertEqual(dls[de.IFM], DataDimLoops(le.IFM, le.BAT))
        self.assertEqual(dls[de.OFM], DataDimLoops(le.OFM, le.BAT))

        clayer = ConvLayer(3, 64, [28, 14], 3, strd=2)
        flayer = FCLayer(2048, 4096, sfil=2)
        self.assertTupleEqual(FCLayer.data_loops(), dls)
        self.assertTupleEqual(clayer.data_loops(), dls)
        self.assertTupleEqual(flayer.data_loops(), dls)

    def test_input_layer(self):
        ''' Get input layer. '''
        clayer = ConvLayer(3, 64, [28, 14], 3, strd=2)
        inlayer = clayer.input_layer()
        self.assertIsInstance(inlayer, Layer)
        self.assertEqual(inlayer.nofm, 3, 'ConvLayer: input_layer: nofm')
        self.assertEqual(inlayer.hofm, (28 - 1) * 2 + 3,
                         'ConvLayer: input_layer: hofm')
        self.assertEqual(inlayer.wofm, (14 - 1) * 2 + 3,
                         'ConvLayer: input_layer: wofm')

    def test_ops(self):
        ''' Get ops. '''
        clayer = ConvLayer(3, 64, [28, 14], 3, strd=2)
        self.assertEqual(clayer.ops_per_neuron(), 3 * 3 * 3,
                         'ConvLayer: ops_per_neurons')
        self.assertEqual(clayer.total_ops(), 3 * 3 * 28 * 14 * 3 * 64,
                         'ConvLayer: total_ops')

    def test_filter_size(self):
        ''' Get filter size. '''
        clayer = ConvLayer(3, 64, [28, 14], 3)
        self.assertEqual(clayer.filter_size(2), 3 * 3 * 2, 'filter_size')
        self.assertEqual(clayer.total_filter_size(2), 3 * 3 * 3 * 64 * 2,
                         'total_filter_size')
        clayer = ConvLayer(3, 64, [28, 14], [3, 1])
        self.assertEqual(clayer.filter_size(2), 3 * 1 * 2, 'filter_size')
        self.assertEqual(clayer.total_filter_size(2), 3 * 1 * 3 * 64 * 2,
                         'total_filter_size')

    def test_filter_size_invalid(self):
        ''' Invalid filter size. '''
        with self.assertRaisesRegex(ValueError, 'ConvLayer: .*sfil.*'):
            _ = ConvLayer(3, 64, [28, 14], [3, 3, 3])

    def test_fclayer(self):
        ''' FCLayer init. '''
        flayer = FCLayer(2048, 4096, sfil=2)
        self.assertEqual(flayer.total_ofmap_size(), 4096, 'FCLayer: ofmap_size')
        self.assertEqual(flayer.filter_size(), 4, 'FCLayer: filter_size')
        self.assertEqual(flayer.total_filter_size(), 2048 * 4096 * 4,
                         'FCLayer: filter_size')

    def test_repr(self):
        ''' __repr__. '''
        # pylint: disable=eval-used
        for l in [ConvLayer(3, 64, [28, 14], [3, 1]),
                  ConvLayer(3, 64, [28, 14], 3, strd=[7, 5]),
                  ConvLayer(3, 64, 28, 3, strd=7), ConvLayer(3, 64, 28, 3)]:
            self.assertIn('ConvLayer', repr(l))
            self.assertEqual(eval(repr(l)), l)

        for l in [FCLayer(2048, 4096),
                  FCLayer(100, 300, 7),
                  FCLayer(100, 300, [7, 3])]:
            self.assertIn('FCLayer', repr(l))
            self.assertEqual(eval(repr(l)), l)


class TestLocalRegionLayer(unittest.TestCase):
    ''' Tests for LocalRegionLayer. '''

    def test_data_loops(self):
        ''' Get data_loops. '''
        dls = LocalRegionLayer.data_loops()
        self.assertEqual(dls[de.FIL], DataDimLoops())
        self.assertEqual(dls[de.IFM], DataDimLoops(le.OFM, le.BAT))
        self.assertEqual(dls[de.OFM], DataDimLoops(le.OFM, le.BAT))

        llayer = LocalRegionLayer(64, 28, 2, 1)
        player = PoolingLayer(64, 28, 2)
        self.assertTupleEqual(PoolingLayer.data_loops(), dls)
        self.assertTupleEqual(llayer.data_loops(), dls)
        self.assertTupleEqual(player.data_loops(), dls)

    def test_ops(self):
        ''' Get ops. '''
        llayer = LocalRegionLayer(64, 28, 2, 1)
        self.assertEqual(llayer.ops_per_neuron(), 2)
        llayer = LocalRegionLayer(64, 28, 1, 3)
        self.assertEqual(llayer.ops_per_neuron(), 9)

    def test_region(self):
        ''' Get region size. '''
        llayer = LocalRegionLayer(64, 28, 2, 1)
        self.assertEqual(llayer.region_size(), 2)
        llayer = LocalRegionLayer(64, 28, 1, [2, 4])
        self.assertEqual(llayer.region_size(), 2 * 4)

    def test_invalid_sreg(self):
        ''' Invalid region size. '''
        with self.assertRaisesRegex(ValueError, 'LocalRegionLayer: .*sreg.*'):
            _ = LocalRegionLayer(64, 28, 1, [2, 4, 6])

    def test_mix_sreg(self):
        ''' Mix region of n-dimension and h/w-dimension. '''
        with self.assertRaisesRegex(ValueError, 'LocalRegionLayer: .*mix.*'):
            _ = LocalRegionLayer(64, 28, 2, 2)

    def test_poolinglayer(self):
        ''' PoolingLayer init. '''
        player = PoolingLayer(64, 28, 2)
        self.assertEqual(player.ops_per_neuron(), 4)
        self.assertEqual(player.total_ifmap_size(),
                         player.total_ofmap_size() * 4)
        player = PoolingLayer(64, 28, 3, strd=2)
        self.assertEqual(player.ops_per_neuron(), 9)

    def test_eltwiselayer(self):
        ''' EltwiseLayer init. '''
        elayer = EltwiseLayer(64, 28, 3)
        self.assertEqual(elayer.ops_per_neuron(), 3)
        self.assertEqual(elayer.nifm, 3 * elayer.nofm)
        self.assertEqual(elayer.ifmap_size(), elayer.ofmap_size())

    def test_repr(self):
        ''' __repr__. '''
        # pylint: disable=eval-used
        for l in [LocalRegionLayer(64, 28, 2, 1),
                  LocalRegionLayer(64, [28, 14], 1, [2, 4]),
                  LocalRegionLayer(64, [28, 14], 1, [2, 4], 7),
                  LocalRegionLayer(64, 28, 1, 4, 7)]:
            self.assertIn('LocalRegionLayer', repr(l))
            self.assertEqual(eval(repr(l)), l)

        for l in [PoolingLayer(64, 28, 2),
                  PoolingLayer(64, 28, 3, strd=2),
                  PoolingLayer(64, [28, 14], [3, 4], strd=[2, 3])]:
            self.assertIn('PoolingLayer', repr(l))
            self.assertEqual(eval(repr(l)), l)

        for l in [EltwiseLayer(64, 32, 3),
                  EltwiseLayer(64, 28, 4)]:
            self.assertIn('EltwiseLayer', repr(l))
            self.assertEqual(eval(repr(l)), l)

