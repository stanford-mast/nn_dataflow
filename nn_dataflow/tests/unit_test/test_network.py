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

from nn_dataflow.core import Network
from nn_dataflow.core import Layer, InputLayer, ConvLayer, FCLayer, \
        PoolingLayer, EltwiseLayer

class TestNetwork(unittest.TestCase):
    ''' Tests for Network. '''
    # pylint: disable=too-many-public-methods

    def setUp(self):
        ''' Set up. '''
        self.network = Network('test_net')
        self.network.set_input_layer(InputLayer(3, 224))
        self.network.add('c1', ConvLayer(3, 64, 224, 3))
        self.network.add('p1', PoolingLayer(64, 7, 32))
        self.network.add('f1', FCLayer(64, 1000, 7))

    def test_set_input_layer(self):
        ''' Modifier set_input_layer. '''
        network = Network('test_net')
        network.set_input_layer(InputLayer(3, 24))
        self.assertIsInstance(network.input_layer(), InputLayer)
        self.assertEqual(network.input_layer().nofm, 3)
        self.assertEqual(network.input_layer().hofm, 24)
        self.assertEqual(network.input_layer().wofm, 24)
        self.assertEqual(len(network), 0)

    def test_set_input_layer_type(self):
        ''' Modifier set_input_layer type. '''
        network = Network('test_net')
        with self.assertRaisesRegex(TypeError, 'Network: .*input_layer.*'):
            network.set_input_layer(Layer(3, 24))
        with self.assertRaisesRegex(TypeError, 'Network: .*input_layer.*'):
            network.set_input_layer(ConvLayer(3, 8, 24, 3))

    def test_set_input_layer_duplicate(self):
        ''' Modifier set_input_layer duplicate. '''
        network = Network('test_net')
        network.set_input_layer(InputLayer(3, 24))
        with self.assertRaisesRegex(KeyError, 'Network: .*input.*'):
            network.set_input_layer(InputLayer(3, 24))

    def test_add(self):
        ''' Modifier add. '''
        self.assertEqual(len(self.network), 3)

        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')
        self.network.add('f3', FCLayer(3000, 1000), prevs=('f1', 'f2'))
        self.network.add('e4', EltwiseLayer(1000, 1, 2), prevs=('f1', 'f3'))
        self.network.add('f4', FCLayer(1000, 1000), prevs='e4')
        self.assertEqual(len(self.network), 7)

    def test_add_same_key(self):
        ''' Modifier add same key. '''
        network = Network('test_net')
        network.set_input_layer(InputLayer(3, 224))

        network.add('c1', ConvLayer(3, 64, 224, 3))
        with self.assertRaisesRegex(KeyError, 'Network: .*c1.*'):
            network.add('c1', ConvLayer(64, 128, 224, 3))

    def test_add_no_input(self):
        ''' Modifier add no input. '''
        network = Network('test_net')

        with self.assertRaisesRegex(RuntimeError, 'Network: .*input.*'):
            network.add('c1', ConvLayer(3, 64, 224, 3))

    def test_add_no_prev(self):
        ''' Modifier add no prevs. '''
        network = Network('test_net')
        network.set_input_layer(InputLayer(3, 224))

        network.add('c1', ConvLayer(3, 64, 224, 3))
        with self.assertRaisesRegex(KeyError, 'Network: .*prev.*p1.*'):
            network.add('p1', PoolingLayer(64, 7, 32), prevs='p1')

    def test_add_invalid_type(self):
        ''' Modifier add invalid type. '''
        network = Network('test_net')
        network.set_input_layer(InputLayer(3, 224))

        with self.assertRaisesRegex(TypeError, 'Network: .*Layer.*'):
            network.add('c1', (3, 64, 224, 3))

    def test_add_unmatch_prev(self):
        ''' Modifier add unmatch prevs. '''
        network = Network('test_net')
        network.set_input_layer(InputLayer(3, 224))
        network.add('c1', ConvLayer(3, 64, 224, 3))

        with self.assertRaisesRegex(ValueError,
                                    'Network: .*c1.*p1.*mismatch fmap.*'):
            network.add('p1', PoolingLayer(64, 7, 2))
        self.assertEqual(len(network), 1)
        with self.assertRaisesRegex(ValueError,
                                    'Network: .*c1.*c2.*mismatch fmap.*'):
            network.add('c2', ConvLayer(64, 128, 220, 3))
        self.assertEqual(len(network), 1)

        with self.assertRaisesRegex(ValueError, 'Network: .*c1.*prev.*p1.*'):
            network.add('p1', PoolingLayer(32, 7, 32))
        self.assertEqual(len(network), 1)
        with self.assertRaisesRegex(ValueError, 'Network: .*c1.*prev.*c2.*'):
            network.add('c2', ConvLayer(32, 128, 224, 3))
        self.assertEqual(len(network), 1)

        network.add('c2', ConvLayer(64, 128, 224, 3))

        with self.assertRaisesRegex(ValueError,
                                    r'Network: .*c1 | c2.*prev.*p1.*'):
            network.add('p1', PoolingLayer(128, 7, 32), prevs=('c1', 'c2'))
        self.assertEqual(len(network), 2)

    def test_add_ext(self):
        ''' Modifier add_ext. '''
        self.assertEqual(len(self.network), 3)

        self.network.add_ext('e0', InputLayer(3, 24))
        self.assertIsInstance(self.network['e0'], InputLayer)
        self.assertEqual(self.network['e0'].nofm, 3)
        self.assertEqual(self.network['e0'].hofm, 24)
        self.assertEqual(self.network['e0'].wofm, 24)

        self.network.add_ext('e1', InputLayer(5, (16, 20)))
        self.assertIsInstance(self.network['e1'], InputLayer)
        self.assertEqual(self.network['e1'].nofm, 5)
        self.assertEqual(self.network['e1'].hofm, 16)
        self.assertEqual(self.network['e1'].wofm, 20)

        self.assertEqual(len(self.network), 3)

    def test_add_ext_same_key(self):
        ''' Modifier add_ext same key. '''
        network = Network('test_net')

        network.add_ext('e0', InputLayer(3, 24))
        with self.assertRaisesRegex(KeyError, 'Network: .*ext.*'):
            network.add_ext('e0', InputLayer(3, 24))

    def test_add_ext_invalid_type(self):
        ''' Modifier add_ext invalid type. '''
        network = Network('test_net')

        with self.assertRaisesRegex(TypeError, 'Network: .*external layer.*'):
            network.add_ext('e0', Layer(3, 24))
        with self.assertRaisesRegex(TypeError, 'Network: .*external layer.*'):
            network.add_ext('e0', ConvLayer(3, 8, 24, 3))

    def test_prevs(self):
        ''' Get prevs. '''
        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')
        self.network.add('f3', FCLayer(3000, 1000), prevs=('f1', 'f2'))

        prevs = self.network.prevs('f1')
        self.assertTupleEqual(prevs, ('p1',))

        prevs = self.network.prevs('f2')
        self.assertTupleEqual(prevs, ('p1',))
        prevs = self.network.prevs('f3')
        self.assertTupleEqual(prevs, ('f1', 'f2'))

    def test_prevs_first(self):
        ''' Get prevs first layer. '''
        self.network.add('c2', ConvLayer(3, 3, 224, 1),
                         prevs=self.network.INPUT_LAYER_KEY)

        prevs = self.network.prevs('c1')
        self.assertTupleEqual(prevs, (None,))

        prevs = self.network.prevs('c2')
        self.assertTupleEqual(prevs, (None,))

    def test_prevs_input(self):
        ''' Get prevs input layer. '''
        with self.assertRaisesRegex(ValueError, 'Network: .*input.*'):
            _ = self.network.prevs(self.network.INPUT_LAYER_KEY)

    def test_prevs_ext_next(self):
        ''' Get prevs next layer of an external layer. '''
        self.network.add_ext('e0', InputLayer(3, 224))

        self.network.add('n', ConvLayer(6, 3, 224, 1),
                         prevs=(self.network.INPUT_LAYER_KEY, 'e0'))

        prevs = self.network.prevs('n')
        self.assertTupleEqual(prevs, (None, 'e0'))

    def test_prevs_ext(self):
        ''' Get prevs external layer. '''
        self.network.add_ext('e0', InputLayer(3, 3))
        with self.assertRaisesRegex(ValueError, 'Network: .*ext.*'):
            _ = self.network.prevs('e0')

    def test_nexts(self):
        ''' Get nexts. '''
        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')
        self.network.add('f3', FCLayer(3000, 1000), prevs=('f1', 'f2'))
        self.network.add('e4', EltwiseLayer(1000, 1, 2), prevs=('f1', 'f3'))
        self.network.add('f4', FCLayer(1000, 1000), prevs='e4')

        nexts = self.network.nexts('p1')
        self.assertTupleEqual(nexts, ('f1', 'f2'))

        nexts = self.network.nexts('f1')
        self.assertTupleEqual(nexts, ('f3', 'e4'))

        nexts = self.network.nexts('f2')
        self.assertTupleEqual(nexts, ('f3',))

        nexts = self.network.nexts('f3')
        self.assertTupleEqual(nexts, ('e4',))

    def test_nexts_last(self):
        ''' Get nexts first layer. '''
        nexts = self.network.nexts('f1')
        self.assertTupleEqual(nexts, (None,))

        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')

        nexts = self.network.nexts('f1')
        self.assertTupleEqual(nexts, (None,))
        nexts = self.network.nexts('f2')
        self.assertTupleEqual(nexts, (None,))

    def test_nexts_input(self):
        ''' Get nexts input layer. '''
        nexts = self.network.nexts(self.network.INPUT_LAYER_KEY)
        self.assertTupleEqual(nexts, ('c1',))

        self.network.add('c2', ConvLayer(3, 3, 224, 1),
                         prevs=self.network.INPUT_LAYER_KEY)
        self.network.add('c3', ConvLayer(6, 4, 224, 1),
                         prevs=(self.network.INPUT_LAYER_KEY, 'c2'))
        nexts = self.network.nexts(self.network.INPUT_LAYER_KEY)
        self.assertTupleEqual(nexts, ('c1', 'c2', 'c3'))

    def test_firsts(self):
        ''' Get firsts. '''
        firsts = self.network.firsts()
        self.assertTupleEqual(firsts, ('c1',))

        self.network.add('c2', ConvLayer(3, 3, 224, 1),
                         prevs=self.network.INPUT_LAYER_KEY)
        self.network.add('c3', ConvLayer(6, 4, 224, 1),
                         prevs=(self.network.INPUT_LAYER_KEY, 'c2'))

        firsts = self.network.firsts()
        self.assertTupleEqual(firsts, ('c1', 'c2'))
        self.assertIn('c1', firsts)
        self.assertNotIn('c3', firsts)

    def test_firsts_ext(self):
        ''' Get firsts with external layers. '''
        self.network.add_ext('e0', InputLayer(3, 224))

        self.network.add('c2', ConvLayer(3, 3, 224, 1), prevs=('e0',))
        self.network.add('c3', ConvLayer(67, 3, 224, 1), prevs=('e0', 'c1'))
        self.network.add('c4', ConvLayer(6, 3, 224, 1),
                         prevs=(self.network.INPUT_LAYER_KEY, 'e0',))

        firsts = self.network.firsts()
        self.assertIn('c2', firsts)
        self.assertNotIn('c3', firsts)
        self.assertIn('c4', firsts)

    def test_lasts(self):
        ''' Get lasts. '''
        lasts = self.network.lasts()
        self.assertTupleEqual(lasts, ('f1',))

        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')

        lasts = self.network.lasts()
        self.assertTupleEqual(lasts, ('f1', 'f2'))

    def test_ext_layers(self):
        ''' Get external layers. '''
        self.assertTupleEqual(self.network.ext_layers(), tuple())

        self.network.add_ext('e0', InputLayer(3, 224))
        self.assertTupleEqual(self.network.ext_layers(), ('e0',))

        self.network.add_ext('e1', InputLayer(3, 224))
        self.assertTupleEqual(self.network.ext_layers(), ('e0', 'e1'))

    def test_contains(self):
        ''' Whether contains. '''
        self.assertIn('c1', self.network)
        self.assertIn('p1', self.network)
        self.assertIn('f1', self.network)
        self.assertNotIn('f2', self.network)

        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')
        self.assertIn('f2', self.network)

    def test_len(self):
        ''' Accessor len. '''
        self.assertEqual(len(self.network), 3)

        network = Network('test_net')
        self.assertEqual(len(network), 0)
        network.set_input_layer(InputLayer(3, 224))
        self.assertEqual(len(network), 0)
        network.add('c1', ConvLayer(3, 4, 224, 1))
        self.assertEqual(len(network), 1)

        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')
        self.assertEqual(len(self.network), 4)
        self.network.add('f3', FCLayer(3000, 1000), prevs=('f1', 'f2'))
        self.assertEqual(len(self.network), 5)
        self.network.add('e4', EltwiseLayer(1000, 1, 2), prevs=('f1', 'f3'))
        self.assertEqual(len(self.network), 6)
        self.network.add('f4', FCLayer(1000, 1000), prevs='e4')
        self.assertEqual(len(self.network), 7)

    def test_iter(self):
        ''' Accessor iter. '''
        num = 0
        for layer in self.network:
            self.assertIn(layer, self.network)
            self.assertIsInstance(self.network[layer], Layer)
            num += 1
        self.assertEqual(len(self.network), num)

        network = Network('test_net')
        network.set_input_layer(InputLayer(3, 224))
        with self.assertRaises(StopIteration):
            _ = next(iter(network))

    def test_contains_ext(self):
        ''' Whether contains external layer. '''
        self.assertNotIn('e0', self.network)
        self.network.add_ext('e0', InputLayer(3, 224))
        self.assertIn('e0', self.network)

    def test_len_ext(self):
        ''' Accessor len external layer. '''
        self.assertEqual(len(self.network), 3)
        self.network.add_ext('e0', InputLayer(3, 224))
        self.assertEqual(len(self.network), 3)

    def test_iter_ext(self):
        ''' Accessor iter external layer. '''
        self.network.add_ext('e0', InputLayer(3, 224))
        for layer in self.network:
            self.assertNotEqual(layer, 'e0')

    def test_getitem(self):
        ''' Accessor getitem. '''
        self.assertIsInstance(self.network['c1'], ConvLayer)
        self.assertIsInstance(self.network['p1'], PoolingLayer)
        self.assertIsInstance(self.network['f1'], FCLayer)

    def test_getitem_error(self):
        ''' Accessor getitem. '''
        with self.assertRaisesRegex(KeyError, 'Network: .*c2.*'):
            _ = self.network['c2']

    def test_str(self):
        ''' Accessor str. '''
        string = str(self.network)
        for layer in self.network:
            self.assertIn(layer, string)

