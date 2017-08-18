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
from collections import OrderedDict

from nn_dataflow.core import partition
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow.core import Network
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import NNDataflowScheme
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import SchedulingResult

class TestNNDataflowScheme(unittest.TestCase):
    ''' Tests for NNDataflowScheme. '''

    def setUp(self):
        self.network = Network('test_net')
        self.network.set_input(InputLayer(3, 224))
        self.network.add('c1', ConvLayer(3, 64, 224, 3))
        self.network.add('p1', PoolingLayer(64, 7, 32), prevs='c1')
        self.network.add('p2', PoolingLayer(64, 7, 32), prevs='c1')
        self.network.add('f1', FCLayer(64, 1000, 7), prevs=['p1', 'p2'])

        self.batch_size = 4

        self.input_layout = partition.get_ofmap_layout(
            self.network.input_layer(), self.batch_size,
            PartitionScheme(order=range(pe.NUM), pdims=[(1, 1)] * pe.NUM),
            NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(2, 1),
                       type=NodeRegion.DATA))

        self.c1res = SchedulingResult(
            dict_loop=OrderedDict([('cost', 1.), ('time', 2.), ('ops', 4.),
                                   ('access', [[7, 8, 9]] * me.NUM),
                                  ]),
            dict_part=OrderedDict([('cost', 0.5), ('total_nhops', [4, 5, 6]),
                                   ('num_nodes', 4),
                                  ]),
            ofmap_layout=partition.get_ofmap_layout(
                self.network['c1'], self.batch_size,
                PartitionScheme(order=range(pe.NUM), pdims=[(1, 1)] * pe.NUM),
                NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 2),
                           type=NodeRegion.DATA)))

        self.pres = SchedulingResult(
            dict_loop=OrderedDict([('cost', 0.1), ('time', 0.05), ('ops', 0.1),
                                   ('access', [[.7, .8, .9]] * me.NUM),
                                  ]),
            dict_part=OrderedDict([('cost', 0.5), ('total_nhops', [.4, .5, .6]),
                                   ('num_nodes', 2),
                                  ]),
            ofmap_layout=partition.get_ofmap_layout(
                self.network['p1'], self.batch_size,
                PartitionScheme(order=range(pe.NUM), pdims=[(1, 1)] * pe.NUM),
                NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 2),
                           type=NodeRegion.DATA)))

        self.dtfl = NNDataflowScheme(self.network, self.input_layout)
        self.dtfl['c1'] = self.c1res
        self.dtfl['p1'] = self.pres
        self.dtfl['p2'] = self.pres

    def test_init(self):
        ''' Initial. '''
        df = NNDataflowScheme(self.network, self.input_layout)

        self.assertEqual(df.network, self.network)
        self.assertEqual(df.input_layout, self.input_layout)

        self.assertEqual(df.total_cost, 0)
        self.assertEqual(df.total_time, 0)
        self.assertFalse(df.res_dict)

        self.assertFalse(df)
        self.assertEqual(df.total_ops, 0)
        self.assertSequenceEqual(df.total_accesses, [0] * me.NUM)
        self.assertEqual(df.total_noc_hops, 0)

    def test_init_invalid_network(self):
        ''' Invalid network. '''
        with self.assertRaisesRegexp(TypeError,
                                     'NNDataflowScheme: .*network*'):
            _ = NNDataflowScheme(self.network['c1'], self.input_layout)

    def test_init_invalid_input_layout(self):
        ''' Invalid input_layout. '''
        with self.assertRaisesRegexp(TypeError,
                                     'NNDataflowScheme: .*input_layout*'):
            _ = NNDataflowScheme(self.network, self.input_layout.frmap)

    def test_setgetitem(self):
        ''' __set/getitem__. '''
        df = NNDataflowScheme(self.network, self.input_layout)

        df['c1'] = self.c1res
        self.assertEqual(df['c1'], self.c1res)

    def test_getitem_not_in(self):
        ''' __getitem__ not in. '''
        df = NNDataflowScheme(self.network, self.input_layout)

        with self.assertRaises(KeyError):
            _ = df['c1']

    def test_setitem_not_in_network(self):
        ''' __setitem__ not in network. '''
        df = NNDataflowScheme(self.network, self.input_layout)

        with self.assertRaisesRegexp(KeyError, 'NNDataflowScheme: .*cc.*'):
            df['cc'] = self.c1res

    def test_setitem_invalid_value(self):
        ''' __setitem__ invalid value. '''
        df = NNDataflowScheme(self.network, self.input_layout)

        with self.assertRaisesRegexp(TypeError,
                                     'NNDataflowScheme: .*SchedulingResult*'):
            df['c1'] = self.c1res.dict_loop

    def test_setitem_already_exists(self):
        ''' __setitem__ already exists. '''
        df = NNDataflowScheme(self.network, self.input_layout)
        df['c1'] = self.c1res

        with self.assertRaisesRegexp(KeyError, 'NNDataflowScheme: .*c1*'):
            df['c1'] = self.c1res

    def test_setitem_prev_not_in(self):
        ''' __setitem__ already exists. '''
        df = NNDataflowScheme(self.network, self.input_layout)

        with self.assertRaisesRegexp(KeyError, 'NNDataflowScheme: .*p1*'):
            df['p1'] = self.pres

    def test_delitem(self):
        ''' __delitem__. '''
        df = NNDataflowScheme(self.network, self.input_layout)
        df['c1'] = self.c1res

        with self.assertRaisesRegexp(KeyError, 'NNDataflowScheme: .*'):
            del df['c1']

    def test_iter_len(self):
        ''' __iter__ and __len__. '''
        self.assertEqual(len(self.dtfl), 3)

        lst = [l for l in self.dtfl]
        self.assertIn('c1', lst)
        self.assertIn('p1', lst)
        self.assertIn('p2', lst)
        self.assertNotIn('f1', lst)

    def test_copy(self):
        ''' copy. '''
        df = self.dtfl
        df2 = df.copy()

        self.assertAlmostEqual(df.total_cost, df2.total_cost)
        self.assertAlmostEqual(df.total_time, df2.total_time)
        self.assertDictEqual(df.res_dict, df2.res_dict)

        # Shallow copy.
        for layer_name in df:
            self.assertEqual(id(df[layer_name]), id(df2[layer_name]))

    def test_properties(self):
        ''' Property accessors. '''
        self.assertAlmostEqual(self.dtfl.total_cost, 1.5 + 0.6 * 2)
        self.assertAlmostEqual(self.dtfl.total_time, 2 + 0.05 * 2)

        self.assertAlmostEqual(self.dtfl.total_ops, 4 + 0.1 * 2)
        for a in self.dtfl.total_accesses:
            self.assertAlmostEqual(a, (7 + 8 + 9) + (.7 + .8 + .9) * 2)
        self.assertAlmostEqual(self.dtfl.total_noc_hops,
                               (4 + 5 + 6) + (.4 + .5 + .6) * 2)
        self.assertAlmostEqual(self.dtfl.total_node_time, 2 * 4 + 0.05 * 2 * 2)

    def test_stats_active_node_pes(self):
        ''' Per-layer stats: active node PEs. '''
        stats = self.dtfl.perlayer_stats('active_node_pes')
        self.assertEqual(len(stats), len(self.dtfl))
        self.assertAlmostEqual(stats['c1'], 0.5)
        self.assertAlmostEqual(stats['p1'], 1)
        self.assertAlmostEqual(stats['p2'], 1)

    def test_stats_total_dram_bw(self):
        ''' Per-layer stats: total DRAM bandwidth. '''
        stats = self.dtfl.perlayer_stats('total_dram_bandwidth')
        self.assertEqual(len(stats), len(self.dtfl))
        self.assertAlmostEqual(stats['c1'], (7 + 8 + 9) / 2.)
        self.assertAlmostEqual(stats['p1'], (.7 + .8 + .9) / 0.05)
        self.assertAlmostEqual(stats['p2'], (.7 + .8 + .9) / 0.05)

    def test_stats_not_supported(self):
        ''' Per-layer stats: not supported. '''
        with self.assertRaisesRegexp(AttributeError,
                                     'NNDataflowScheme: .*not_supported.*'):
            _ = self.dtfl.perlayer_stats('not_supported')

