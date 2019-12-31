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
from collections import OrderedDict

from nn_dataflow.core import DataLayout
from nn_dataflow.core import FmapPosition, FmapRange
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
    # pylint: disable=too-many-public-methods

    # pylint: disable=too-many-public-methods

    def setUp(self):
        self.network = Network('test_net')
        self.network.set_input_layer(InputLayer(3, 224))
        self.network.add('c1', ConvLayer(3, 64, 224, 3))
        self.network.add('p1', PoolingLayer(64, 7, 32), prevs='c1')
        self.network.add('p2', PoolingLayer(64, 7, 32), prevs='c1')
        self.network.add('f1', FCLayer(128, 1000, 7), prevs=['p1', 'p2'])

        self.batch_size = 4

        input_layer = self.network.input_layer()
        self.input_layout = DataLayout(
            frngs=(FmapRange((0, 0, 0, 0),
                             FmapPosition(b=self.batch_size,
                                          n=input_layer.nofm,
                                          h=input_layer.hofm,
                                          w=input_layer.wofm)),),
            regions=(NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(2, 1),
                                type=NodeRegion.DRAM),),
            parts=(PartitionScheme(order=range(pe.NUM),
                                   pdims=[(1, 1)] * pe.NUM),))

        c1_layer = self.network['c1']
        self.c1res = SchedulingResult(
            scheme=OrderedDict([('cost', 1.5), ('time', 200.), ('ops', 4.),
                                ('num_nodes', 4),
                                ('cost_op', 0.5), ('cost_access', 1.),
                                ('cost_noc', 0), ('cost_static', 0),
                                ('proc_time', 200), ('bus_time', 0),
                                ('dram_time', 0),
                                ('access', [[7, 8, 9]] * me.NUM),
                                ('remote_gbuf_access', [0] * 3),
                                ('total_nhops', [4, 5, 6]),
                                ('fetch', [[1, 1, 1], [2, 2, 2]]),
                                ('ti', [2, 2, 3]),
                                ('to', [1, 2, 3]),
                                ('tb', [1, 2, 3]),
                                ('tvals', [[2, 1, 1], [2, 2, 2], [3, 3, 3]]),
                                ('orders', [range(3)] * 2),
                               ]),
            ofmap_layout=DataLayout(
                frngs=(FmapRange((0, 0, 0, 0),
                                 FmapPosition(b=self.batch_size,
                                              n=c1_layer.nofm,
                                              h=c1_layer.hofm,
                                              w=c1_layer.wofm)),),
                regions=(NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 2),
                                    type=NodeRegion.DRAM),),
                parts=(PartitionScheme(order=range(pe.NUM),
                                       pdims=[(1, 1)] * pe.NUM),)),
            sched_seq=(0, 0, 0))

        p1_layer = self.network['p1']
        self.p1res = SchedulingResult(
            scheme=OrderedDict([('cost', 0.6), ('time', 5), ('ops', 0.1),
                                ('num_nodes', 2),
                                ('cost_op', 0.1), ('cost_access', 0.5),
                                ('cost_noc', 0), ('cost_static', 0),
                                ('proc_time', 5), ('bus_time', 0),
                                ('dram_time', 0),
                                ('access', [[.7, .8, .9]] * me.NUM),
                                ('remote_gbuf_access', [0] * 3),
                                ('total_nhops', [.4, .5, .6]),
                                ('fetch', [[1, 1, 1], [2, 2, 2]]),
                                ('ti', [2, 2, 3]),
                                ('to', [1, 2, 3]),
                                ('tb', [1, 2, 3]),
                                ('tvals', [[2, 1, 1], [2, 2, 2], [3, 3, 3]]),
                                ('orders', [range(3)] * 2),
                               ]),
            ofmap_layout=DataLayout(
                frngs=(FmapRange((0, 0, 0, 0),
                                 FmapPosition(b=self.batch_size,
                                              n=p1_layer.nofm,
                                              h=p1_layer.hofm,
                                              w=p1_layer.wofm)),),
                regions=(NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 2),
                                    type=NodeRegion.DRAM),),
                parts=(PartitionScheme(order=range(pe.NUM),
                                       pdims=[(1, 1)] * pe.NUM),)),
            sched_seq=(0, 1, 0))

        self.p2res = SchedulingResult(
            scheme=self.p1res.scheme, ofmap_layout=self.p1res.ofmap_layout,
            sched_seq=(0, 2, 0))

        self.dtfl = NNDataflowScheme(self.network, self.input_layout)
        self.dtfl['c1'] = self.c1res
        self.dtfl['p1'] = self.p1res
        self.dtfl['p2'] = self.p2res

    def test_init(self):
        ''' Initial. '''
        df = NNDataflowScheme(self.network, self.input_layout)

        self.assertEqual(df.network, self.network)
        self.assertEqual(df.input_layout, self.input_layout)
        self.assertDictEqual(df.ext_layout_dict, {})

        self.assertEqual(df.total_cost, 0)
        self.assertEqual(df.total_time, 0)
        self.assertFalse(df.res_dict)

        self.assertFalse(df)
        self.assertEqual(df.total_ops, 0)
        self.assertSequenceEqual(df.total_accesses, [0] * me.NUM)
        self.assertEqual(df.total_noc_hops, 0)

    def test_init_ext(self):
        ''' Initial with external layers. '''
        self.network.add_ext('e0', InputLayer(3, 224))
        self.network.add_ext('e1', InputLayer(6, 224))

        e0_layout = DataLayout(
            frngs=(FmapRange((0, 0, 0, 0),
                             FmapPosition(b=self.batch_size,
                                          n=self.network['e0'].nofm,
                                          h=self.network['e0'].hofm,
                                          w=self.network['e0'].wofm)),),
            regions=self.input_layout.regions,
            parts=self.input_layout.parts)
        e1_layout = DataLayout(
            frngs=(FmapRange((0, 0, 0, 0),
                             FmapPosition(b=self.batch_size,
                                          n=self.network['e1'].nofm,
                                          h=self.network['e1'].hofm,
                                          w=self.network['e1'].wofm)),),
            regions=self.input_layout.regions,
            parts=self.input_layout.parts)

        ext_layout_dict = {'e0': e0_layout, 'e1': e1_layout}

        df = NNDataflowScheme(self.network, self.input_layout,
                              ext_layout_dict)

        self.assertIn('e0', df.ext_layout_dict)
        self.assertIn('e1', df.ext_layout_dict)
        self.assertEqual(df.ext_layout_dict['e0'], e0_layout)
        self.assertEqual(df.ext_layout_dict['e1'], e1_layout)

    def test_init_invalid_network(self):
        ''' Invalid network. '''
        with self.assertRaisesRegex(TypeError,
                                    'NNDataflowScheme: .*network*'):
            _ = NNDataflowScheme(self.network['c1'], self.input_layout)

    def test_init_invalid_input_layout(self):
        ''' Invalid input_layout. '''
        with self.assertRaisesRegex(TypeError,
                                    'NNDataflowScheme: .*input_layout*'):
            _ = NNDataflowScheme(self.network, self.input_layout.frngs)

    def test_init_invalid_eld_keys(self):
        ''' Invalid ext_layout_dict keys. '''
        with self.assertRaisesRegex(ValueError,
                                    'NNDataflowScheme: .*ext_layout_dict*'):
            _ = NNDataflowScheme(self.network, self.input_layout,
                                 {'e0': self.input_layout})

        self.network.add_ext('e0', InputLayer(3, 224))
        with self.assertRaisesRegex(ValueError,
                                    'NNDataflowScheme: .*ext_layout_dict*'):
            _ = NNDataflowScheme(self.network, self.input_layout)

    def test_init_invalid_eld_type(self):
        ''' Invalid ext_layout_dict value type. '''
        self.network.add_ext('e0', InputLayer(3, 224))
        self.network.add_ext('e1', InputLayer(3, 224))

        with self.assertRaisesRegex(TypeError,
                                    'NNDataflowScheme: .*ext_layout*'):
            _ = NNDataflowScheme(self.network, self.input_layout,
                                 {'e0': self.input_layout,
                                  'e1': self.input_layout.frngs})

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

        with self.assertRaisesRegex(KeyError, 'NNDataflowScheme: .*cc.*'):
            df['cc'] = self.c1res

    def test_setitem_invalid_value(self):
        ''' __setitem__ invalid value. '''
        df = NNDataflowScheme(self.network, self.input_layout)

        with self.assertRaisesRegex(TypeError,
                                    'NNDataflowScheme: .*SchedulingResult*'):
            df['c1'] = self.c1res.scheme

    def test_setitem_already_exists(self):
        ''' __setitem__ already exists. '''
        df = NNDataflowScheme(self.network, self.input_layout)
        df['c1'] = self.c1res

        with self.assertRaisesRegex(KeyError, 'NNDataflowScheme: .*c1*'):
            df['c1'] = self.c1res._replace(sched_seq=(1, 0, 0))

    def test_setitem_prev_not_in(self):
        ''' __setitem__ previous not existing. '''
        df = NNDataflowScheme(self.network, self.input_layout)

        with self.assertRaisesRegex(KeyError, 'NNDataflowScheme: .*p1*'):
            df['p1'] = self.p1res

    def test_setitem_prev_input_ext(self):
        ''' __setitem__ previous is input or external. '''
        df = NNDataflowScheme(self.network, self.input_layout)
        df['c1'] = self.c1res
        self.assertAlmostEqual(df.total_cost, self.c1res.total_cost)

        self.network.add_ext('e0', InputLayer(3, 224))
        self.network.add('c2', self.network['c1'], prevs=('e0',))
        df = NNDataflowScheme(self.network, self.input_layout,
                              {'e0': self.input_layout})
        df['c2'] = self.c1res
        self.assertAlmostEqual(df.total_cost, self.c1res.total_cost)

    def test_setitem_invalid_seg_idx(self):
        ''' __setitem__ invalid segment index. '''
        df = NNDataflowScheme(self.network, self.input_layout)

        with self.assertRaisesRegex(ValueError,
                                    'NNDataflowScheme: .*segment index*'):
            df['c1'] = self.c1res._replace(sched_seq=(1, 0, 0))

        df = NNDataflowScheme(self.network, self.input_layout)
        df['c1'] = self.c1res
        df['p1'] = self.p1res._replace(sched_seq=(1, 0, 0))

        with self.assertRaisesRegex(ValueError,
                                    'NNDataflowScheme: .*segment index*'):
            df['p2'] = self.p2res._replace(sched_seq=(0, 0, 0))

    def test_delitem(self):
        ''' __delitem__. '''
        df = NNDataflowScheme(self.network, self.input_layout)
        df['c1'] = self.c1res

        with self.assertRaisesRegex(KeyError, 'NNDataflowScheme: .*'):
            del df['c1']

    def test_iter_len(self):
        ''' __iter__ and __len__. '''
        self.assertEqual(len(self.dtfl), 3)

        lst = [l for l in self.dtfl]  # pylint: disable=unnecessary-comprehension
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

    def test_copy_ext(self):
        ''' copy external layers. '''
        self.network.add_ext('e0', self.network.input_layer())
        self.network.add_ext('e1', self.network.input_layer())

        df1 = NNDataflowScheme(self.network, self.input_layout,
                               {'e0': self.input_layout,
                                'e1': self.input_layout})
        df1['c1'] = self.c1res
        df1['p1'] = self.p1res
        df1['p2'] = self.p2res

        df2 = df1.copy()

        self.assertAlmostEqual(df1.total_cost, df2.total_cost)
        self.assertAlmostEqual(df1.total_time, df2.total_time)
        self.assertDictEqual(df1.res_dict, df2.res_dict)
        self.assertDictEqual(df1.ext_layout_dict, df2.ext_layout_dict)

    def test_fmap_layout(self):
        ''' fmap_layout. '''
        flayout = self.dtfl.fmap_layout(('c1',))
        frng = flayout.complete_fmap_range()
        self.assertTrue(flayout.is_in(self.c1res.ofmap_layout.regions[0]))
        self.assertEqual(frng, self.c1res.ofmap_layout.frngs[0])

        flayout = self.dtfl.fmap_layout((None,))
        frng = flayout.complete_fmap_range()
        self.assertTrue(flayout.is_in(self.input_layout.regions[0]))
        self.assertEqual(frng, self.input_layout.frngs[0])

        flayout = self.dtfl.fmap_layout(('p1', 'p2'))
        frng = flayout.complete_fmap_range()
        self.assertEqual(frng.size('n'),
                         self.network['p1'].nofm + self.network['p2'].nofm)

        flayout = self.dtfl.fmap_layout((None, 'c1'))
        frng = flayout.complete_fmap_range()
        self.assertEqual(frng.size('n'),
                         self.network.input_layer().nofm
                         + self.network['c1'].nofm)

    def test_fmap_layout_ext(self):
        ''' fmap_layout external layers. '''
        self.network.add_ext('e0', self.network.input_layer())
        self.network.add_ext('e1', self.network.input_layer())

        df = NNDataflowScheme(self.network, self.input_layout,
                              {'e0': self.input_layout,
                               'e1': self.input_layout})
        df['c1'] = self.c1res
        df['p1'] = self.p1res
        df['p2'] = self.p2res

        flayout = df.fmap_layout(('e0',))
        self.assertEqual(flayout, self.input_layout)

        flayout = df.fmap_layout(('e1', None))
        self.assertTrue(flayout.is_in(self.input_layout.regions[0]))
        frng = flayout.complete_fmap_range()
        self.assertEqual(frng.size('n'),
                         self.network['e1'].nofm
                         + self.network.input_layer().nofm)

    def test_properties(self):
        ''' Property accessors. '''
        self.assertAlmostEqual(self.dtfl.total_cost, 1.5 + 0.6 * 2)
        self.assertAlmostEqual(self.dtfl.total_time, 200 + 5)

        self.assertAlmostEqual(self.dtfl.total_ops, 4 + 0.1 * 2)
        for a in self.dtfl.total_accesses:
            self.assertAlmostEqual(a, (7 + 8 + 9) + (.7 + .8 + .9) * 2)
        self.assertAlmostEqual(self.dtfl.total_noc_hops,
                               (4 + 5 + 6) + (.4 + .5 + .6) * 2)

    def test_time_full_net_single_seg(self):
        ''' time() when full network fits in a single segment. '''
        dtfl = NNDataflowScheme(self.network, self.input_layout)
        dtfl['c1'] = self.c1res
        dtfl['p1'] = self.p1res._replace(sched_seq=(0, 1, 0))
        dtfl['p2'] = self.p2res._replace(sched_seq=(0, 2, 0))
        dtfl['f1'] = self.c1res._replace(sched_seq=(0, 3, 0))
        self.assertEqual(dtfl.total_time, 200)

    def test_static_cost_adjust(self):
        ''' Adjust static cost portion. '''

        # Add static cost.
        idl_unit_cost = 1e-3

        c1scheme = self.c1res.scheme
        c1static = c1scheme['time'] * idl_unit_cost
        c1scheme['cost_static'] += c1static
        c1scheme['cost_access'] -= c1static

        p1scheme = self.p1res.scheme
        p1static = p1scheme['time'] * idl_unit_cost
        p1scheme['cost_static'] += p1static
        p1scheme['cost_access'] -= p1static

        # No adjust.
        dtfl = NNDataflowScheme(self.network, self.input_layout)
        dtfl['c1'] = self.c1res._replace(scheme=c1scheme)
        dtfl['p1'] = self.p1res._replace(scheme=p1scheme, sched_seq=(1, 0, 0))
        dtfl['p2'] = self.p2res._replace(scheme=p1scheme, sched_seq=(2, 0, 0))
        dtfl['f1'] = self.c1res._replace(scheme=c1scheme, sched_seq=(3, 0, 0))

        sum_cost = 1.5 + 0.6 + 0.6 + 1.5
        sum_time = 200 + 5 + 5 + 200

        self.assertAlmostEqual(dtfl.total_cost, sum_cost)
        self.assertAlmostEqual(dtfl.total_time, sum_time)

        # With adjust.
        dtfl = NNDataflowScheme(self.network, self.input_layout)
        dtfl['c1'] = self.c1res._replace(scheme=c1scheme)
        dtfl['p1'] = self.p1res._replace(scheme=p1scheme, sched_seq=(0, 1, 0))
        dtfl['p2'] = self.p2res._replace(scheme=p1scheme, sched_seq=(0, 2, 0))
        dtfl['f1'] = self.c1res._replace(scheme=c1scheme, sched_seq=(1, 0, 0))

        diff = (sum_time - dtfl.total_time) * idl_unit_cost
        self.assertGreater(diff, 0)
        self.assertAlmostEqual(dtfl.total_cost, sum_cost -diff)

        # All in one segment.
        dtfl = NNDataflowScheme(self.network, self.input_layout)
        dtfl['c1'] = self.c1res._replace(scheme=c1scheme)
        dtfl['p1'] = self.p1res._replace(scheme=p1scheme, sched_seq=(0, 1, 0))
        dtfl['p2'] = self.p2res._replace(scheme=p1scheme, sched_seq=(0, 2, 0))
        dtfl['f1'] = self.c1res._replace(scheme=c1scheme, sched_seq=(0, 3, 0))

        diff = (sum_time - dtfl.total_time) * idl_unit_cost
        self.assertGreater(diff, 0)
        self.assertAlmostEqual(dtfl.total_cost, sum_cost -diff)

    def test_segment_time_list(self):
        ''' segment_time_list(). '''
        dtfl = NNDataflowScheme(self.network, self.input_layout)
        dtfl['c1'] = self.c1res
        dtfl['p1'] = self.p1res
        dtfl['p2'] = self.p2res._replace(sched_seq=(1, 0, 0))
        self.assertListEqual(dtfl.segment_time_list(), [205, 5])

    def test_segment_dram_time_list(self):
        ''' segment_dram_time_list(). '''
        c1_scheme = self.c1res.scheme.copy()
        c1_scheme['dram_time'] = 180
        p1_scheme = self.p1res.scheme.copy()
        p1_scheme['dram_time'] = 5
        p2_scheme = self.p2res.scheme.copy()
        p2_scheme['dram_time'] = 10
        dtfl = NNDataflowScheme(self.network, self.input_layout)
        dtfl['c1'] = self.c1res._replace(scheme=c1_scheme)
        dtfl['p1'] = self.p1res._replace(scheme=p1_scheme)
        dtfl['p2'] = self.p2res._replace(sched_seq=(1, 0, 0),
                                         scheme=p2_scheme)
        self.assertListEqual(dtfl.segment_dram_time_list(), [185, 10])
        self.assertListEqual(dtfl.segment_time_list(), [205, 10])

    def test_stats_active_node_pes(self):
        ''' Per-layer stats: active node PEs. '''
        stats = self.dtfl.perlayer_stats('active_node_pes')
        self.assertEqual(len(stats), len(self.dtfl))
        self.assertAlmostEqual(stats['c1'], 0.005)
        self.assertAlmostEqual(stats['p1'], 0.01)
        self.assertAlmostEqual(stats['p2'], 0.01)

    def test_stats_dram_bandwidth(self):
        ''' Per-layer stats: DRAM bandwidth. '''
        stats = self.dtfl.perlayer_stats('dram_bandwidth')
        self.assertEqual(len(stats), len(self.dtfl))
        self.assertAlmostEqual(stats['c1'], (7. + 8. + 9.) / 200)
        self.assertAlmostEqual(stats['p1'], (.7 + .8 + .9) / 5)
        self.assertAlmostEqual(stats['p2'], (.7 + .8 + .9) / 5)

    def test_stats_not_supported(self):
        ''' Per-layer stats: not supported. '''
        with self.assertRaisesRegex(AttributeError,
                                    'NNDataflowScheme: .*not_supported.*'):
            _ = self.dtfl.perlayer_stats('not_supported')

