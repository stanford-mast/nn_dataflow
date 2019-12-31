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
from nn_dataflow.core import InputLayer

import nn_dataflow.nns as nns

class TestNNs(unittest.TestCase):
    ''' Tests for NN definitions. '''

    def test_all_networks(self):
        ''' Get all_networks. '''
        self.assertIn('alex_net', nns.all_networks())
        self.assertIn('vgg_net', nns.all_networks())
        self.assertGreater(len(nns.all_networks()), 5)

    def test_import_network(self):
        ''' Get import_network. '''
        for name in nns.all_networks():
            network = nns.import_network(name)
            self.assertIsInstance(network, Network)

    def test_import_network_invalid(self):
        ''' Get import_network invalid. '''
        with self.assertRaisesRegex(ImportError, 'nns: .*defined.*'):
            _ = nns.import_network('aaa')

    def test_add_lstm_cell(self):
        ''' Add LSTM cell. '''
        net = Network('LSTM')
        net.set_input_layer(InputLayer(512, 1))
        c, h = nns.add_lstm_cell(net, 'cell0', 512,
                                 net.INPUT_LAYER_KEY, net.INPUT_LAYER_KEY,
                                 net.INPUT_LAYER_KEY)
        c, h = nns.add_lstm_cell(net, 'cell1', 512,
                                 net.INPUT_LAYER_KEY, c, h)
        c, h = nns.add_lstm_cell(net, 'cell2', 512,
                                 net.INPUT_LAYER_KEY, c, h)
        num_weights = 0
        for layer in net:
            try:
                num_weights += net[layer].total_filter_size()
            except AttributeError:
                pass
        self.assertEqual(num_weights, 512 * 512 * 2 * 4 * 3)

    def test_add_lstm_cell_invalid_type(self):
        ''' Add LSTM cell with invalid type. '''
        with self.assertRaisesRegex(TypeError, 'add_lstm_cell: .*network.*'):
            _ = nns.add_lstm_cell(InputLayer(512, 1), 'cell0', 512,
                                  None, None, None)

    def test_add_lstm_cell_not_in(self):
        ''' Add LSTM cell input not in. '''
        net = Network('LSTM')
        net.set_input_layer(InputLayer(512, 1))
        with self.assertRaisesRegex(ValueError, 'add_lstm_cell: .*in.*'):
            _ = nns.add_lstm_cell(net, 'cell0', 512,
                                  'a', net.INPUT_LAYER_KEY,
                                  net.INPUT_LAYER_KEY)

        net = Network('LSTM')
        net.set_input_layer(InputLayer(512, 1))
        with self.assertRaisesRegex(ValueError, 'add_lstm_cell: .*in.*'):
            _ = nns.add_lstm_cell(net, 'cell0', 512,
                                  net.INPUT_LAYER_KEY, 'a',
                                  net.INPUT_LAYER_KEY)

        net = Network('LSTM')
        net.set_input_layer(InputLayer(512, 1))
        with self.assertRaisesRegex(ValueError, 'add_lstm_cell: .*in.*'):
            _ = nns.add_lstm_cell(net, 'cell0', 512,
                                  net.INPUT_LAYER_KEY, net.INPUT_LAYER_KEY,
                                  'a')

