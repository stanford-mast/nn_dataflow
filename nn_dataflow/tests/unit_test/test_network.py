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

from nn_dataflow.core import Network
from nn_dataflow.core import Layer, InputLayer, ConvLayer, FCLayer, PoolingLayer

class TestNetwork(unittest.TestCase):
    ''' Tests for Network. '''
    # pylint: disable=too-many-public-methods

    def setUp(self):
        ''' Set up. '''
        self.network = Network('test_net')
        self.network.set_input(InputLayer(3, 224))
        self.network.add('c1', ConvLayer(3, 64, 224, 3))
        self.network.add('p1', PoolingLayer(64, 7, 32))
        self.network.add('f1', FCLayer(64, 1000, 7))

    def test_set_input(self):
        ''' Modifier set_input. '''
        network = Network('test_net')
        network.set_input(InputLayer(3, 24))
        self.assertIsInstance(network.input_layer(), InputLayer)
        self.assertEqual(network.input_layer().nofm, 3)
        self.assertEqual(network.input_layer().hofm, 24)
        self.assertEqual(network.input_layer().wofm, 24)
        self.assertEqual(len(network), 0)

    def test_set_input_type(self):
        ''' Modifier set_input type. '''
        network = Network('test_net')
        with self.assertRaisesRegexp(TypeError, 'Network: .*input_layer.*'):
            network.set_input(Layer(3, 24))
        with self.assertRaisesRegexp(TypeError, 'Network: .*input_layer.*'):
            network.set_input(ConvLayer(3, 8, 24, 3))

    def test_set_input_duplicate(self):
        ''' Modifier set_input duplicate. '''
        network = Network('test_net')
        network.set_input(InputLayer(3, 24))
        with self.assertRaisesRegexp(KeyError, 'Network: .*input.*'):
            network.set_input(InputLayer(3, 24))

    def test_add(self):
        ''' Modifier add. '''
        self.assertEqual(len(self.network), 3)

        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')
        self.network.add('f3', FCLayer(3000, 1000), prevs=('f1', 'f2'))
        self.network.add('f4', FCLayer(1000, 1000), prevs=('f1', 'f3'))
        self.assertEqual(len(self.network), 6)

    def test_add_same_key(self):
        ''' Modifier add same key. '''
        network = Network('test_net')
        network.set_input(InputLayer(3, 224))

        network.add('c1', ConvLayer(3, 64, 224, 3))
        with self.assertRaisesRegexp(KeyError, 'Network: .*c1.*'):
            network.add('c1', ConvLayer(64, 128, 224, 3))

    def test_add_no_input(self):
        ''' Modifier add no input. '''
        network = Network('test_net')

        with self.assertRaisesRegexp(RuntimeError, 'Network: .*input.*'):
            network.add('c1', ConvLayer(3, 64, 224, 3))

    def test_add_no_prev(self):
        ''' Modifier add no prevs. '''
        network = Network('test_net')
        network.set_input(InputLayer(3, 224))

        network.add('c1', ConvLayer(3, 64, 224, 3))
        with self.assertRaisesRegexp(KeyError, 'Network: .*prev.*p1.*'):
            network.add('p1', PoolingLayer(64, 7, 32), prevs='p1')

    def test_add_invalid_type(self):
        ''' Modifier add invalid type. '''
        network = Network('test_net')
        network.set_input(InputLayer(3, 224))

        with self.assertRaisesRegexp(TypeError, 'Network: .*Layer.*'):
            network.add('c1', (3, 64, 224, 3))

    def test_add_unmatch_prev(self):
        ''' Modifier add unmatch prevs. '''
        network = Network('test_net')
        network.set_input(InputLayer(3, 224))
        network.add('c1', ConvLayer(3, 64, 224, 3))

        with self.assertRaisesRegexp(ValueError,
                                     'Network: .*c1.*p1.*mismatch fmap.*'):
            network.add('p1', PoolingLayer(64, 7, 2))
        self.assertEqual(len(network), 1)
        with self.assertRaisesRegexp(ValueError,
                                     'Network: .*c1.*c2.*mismatch fmap.*'):
            network.add('c2', ConvLayer(64, 128, 220, 3))
        self.assertEqual(len(network), 1)

        with self.assertRaisesRegexp(ValueError, 'Network: .*merge.*c1.*p1.*'):
            network.add('p1', PoolingLayer(32, 7, 32))
        self.assertEqual(len(network), 1)
        with self.assertRaisesRegexp(ValueError, 'Network: .*merge.*c1.*c2.*'):
            network.add('c2', ConvLayer(32, 128, 224, 3))
        self.assertEqual(len(network), 1)

        network.add('c2', ConvLayer(64, 128, 224, 3))

        with self.assertRaisesRegexp(ValueError,
                                     r'Network: .*merge.*c1\s*c2.*p1.*'):
            network.add('p1', PoolingLayer(128, 7, 32), prevs=('c1', 'c2'))
        self.assertEqual(len(network), 2)

    def test_prev_layers(self):
        ''' Get prev_layers. '''
        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')
        self.network.add('f3', FCLayer(3000, 1000), prevs=('f1', 'f2'))
        self.network.add('f4', FCLayer(1000, 1000), prevs=('f1', 'f3'))

        prevs, symbol = self.network.prev_layers('f1')
        self.assertTupleEqual(prevs, ('p1',))
        self.assertEqual(symbol, '|')

        prevs, symbol = self.network.prev_layers('f2')
        self.assertTupleEqual(prevs, ('p1',))
        self.assertEqual(symbol, '|')
        prevs, symbol = self.network.prev_layers('f3')
        self.assertTupleEqual(prevs, ('f1', 'f2'))
        self.assertEqual(symbol, '|')

        prevs, symbol = self.network.prev_layers('f4')
        self.assertTupleEqual(prevs, ('f1', 'f3'))
        self.assertEqual(symbol, '+')

    def test_prev_layers_first(self):
        ''' Get prev_layers first layer. '''
        self.network.add('c2', ConvLayer(3, 3, 224, 1),
                         prevs=self.network.INPUT_LAYER_KEY)
        self.network.add('c3', ConvLayer(3, 4, 224, 1),
                         prevs=(self.network.INPUT_LAYER_KEY, 'c2'))

        prevs, symbol = self.network.prev_layers('c1')
        self.assertTupleEqual(prevs, (None,))
        self.assertEqual(symbol, '|')

        prevs, symbol = self.network.prev_layers('c2')
        self.assertTupleEqual(prevs, (None,))
        self.assertEqual(symbol, '|')

        prevs, symbol = self.network.prev_layers('c3')
        self.assertTupleEqual(prevs, (None, 'c2'))
        self.assertEqual(symbol, '+')

    def test_prev_layers_input(self):
        ''' Get prev_layers input layer. '''
        with self.assertRaisesRegexp(ValueError, 'Network: .*input.*'):
            _ = self.network.prev_layers(self.network.INPUT_LAYER_KEY)

    def test_next_layers(self):
        ''' Get next_layers. '''
        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')
        self.network.add('f3', FCLayer(3000, 1000), prevs=('f1', 'f2'))
        self.network.add('f4', FCLayer(1000, 1000), prevs=('f1', 'f3'))

        nexts = self.network.next_layers('p1')
        self.assertTupleEqual(nexts, ('f1', 'f2'))

        nexts = self.network.next_layers('f1')
        self.assertTupleEqual(nexts, ('f3', 'f4'))

        nexts = self.network.next_layers('f2')
        self.assertTupleEqual(nexts, ('f3',))

        nexts = self.network.next_layers('f3')
        self.assertTupleEqual(nexts, ('f4',))

    def test_next_layers_last(self):
        ''' Get next_layers first layer. '''
        nexts = self.network.next_layers('f1')
        self.assertTupleEqual(nexts, (None,))

        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')

        nexts = self.network.next_layers('f1')
        self.assertTupleEqual(nexts, (None,))
        nexts = self.network.next_layers('f2')
        self.assertTupleEqual(nexts, (None,))

    def test_next_layers_input(self):
        ''' Get next_layers input layer. '''
        nexts = self.network.next_layers(self.network.INPUT_LAYER_KEY)
        self.assertTupleEqual(nexts, ('c1',))

        self.network.add('c2', ConvLayer(3, 3, 224, 1),
                         prevs=self.network.INPUT_LAYER_KEY)
        self.network.add('c3', ConvLayer(3, 4, 224, 1),
                         prevs=(self.network.INPUT_LAYER_KEY, 'c2'))
        nexts = self.network.next_layers(self.network.INPUT_LAYER_KEY)
        self.assertTupleEqual(nexts, ('c1', 'c2', 'c3'))

    def test_first_layers(self):
        ''' Get first_layers. '''
        firsts = self.network.first_layers()
        self.assertTupleEqual(firsts, ('c1',))

        self.network.add('c2', ConvLayer(3, 3, 224, 1),
                         prevs=self.network.INPUT_LAYER_KEY)
        self.network.add('c3', ConvLayer(3, 4, 224, 1),
                         prevs=(self.network.INPUT_LAYER_KEY, 'c2'))

        firsts = self.network.first_layers()
        self.assertTupleEqual(firsts, ('c1', 'c2'))
        self.assertIn('c1', firsts)
        self.assertNotIn('c3', firsts)

    def test_last_layers(self):
        ''' Get last_layers. '''
        lasts = self.network.last_layers()
        self.assertTupleEqual(lasts, ('f1',))

        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')

        lasts = self.network.last_layers()
        self.assertTupleEqual(lasts, ('f1', 'f2'))

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
        network.set_input(InputLayer(3, 224))
        self.assertEqual(len(network), 0)
        network.add('c1', ConvLayer(3, 4, 224, 1))
        self.assertEqual(len(network), 1)

        self.network.add('f2', FCLayer(64, 2000, 7), prevs='p1')
        self.assertEqual(len(self.network), 4)
        self.network.add('f3', FCLayer(3000, 1000), prevs=('f1', 'f2'))
        self.assertEqual(len(self.network), 5)
        self.network.add('f4', FCLayer(1000, 1000), prevs=('f1', 'f3'))
        self.assertEqual(len(self.network), 6)

    def test_iter(self):
        ''' Accessor iter. '''
        num = 0
        for layer in self.network:
            self.assertIn(layer, self.network)
            self.assertIsInstance(self.network[layer], Layer)
            num += 1
        self.assertEqual(len(self.network), num)

        network = Network('test_net')
        network.set_input(InputLayer(3, 224))
        with self.assertRaises(StopIteration):
            _ = next(iter(network))

    def test_getitem(self):
        ''' Accessor getitem. '''
        self.assertIsInstance(self.network['c1'], ConvLayer)
        self.assertIsInstance(self.network['p1'], PoolingLayer)
        self.assertIsInstance(self.network['f1'], FCLayer)

    def test_getitem_error(self):
        ''' Accessor getitem. '''
        with self.assertRaisesRegexp(KeyError, 'Network: .*c2.*'):
            _ = self.network['c2']

    def test_str(self):
        ''' Accessor str. '''
        string = str(self.network)
        for layer in self.network:
            self.assertIn(layer, string)

