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
        with self.assertRaisesRegexp(ImportError, 'nns: .*defined.*'):
            _ = nns.import_network('aaa')

