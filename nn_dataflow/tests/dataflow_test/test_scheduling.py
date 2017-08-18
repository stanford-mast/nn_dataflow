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

from nn_dataflow.core import partition
from nn_dataflow.core import ConvLayer, LocalRegionLayer, PoolingLayer
from nn_dataflow.core import Cost
from nn_dataflow.core import MapStrategyEyeriss
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import Option
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource
from nn_dataflow.core import Scheduling
from nn_dataflow.core import SchedulingCondition, SchedulingResult

class TestScheduling(unittest.TestCase):
    ''' Tests for Scheduling module. '''

    def setUp(self):

        self.layers = {}
        self.layers['BASE'] = ConvLayer(8, 16, 28, 3)
        self.layers['POOL'] = PoolingLayer(16, 28, 2)
        self.layers['LR'] = LocalRegionLayer(16, 28, nreg=3, sreg=1)

        self.batch_size = 4

        self.cost = Cost(mac_op=1, mem_hier=(200, 6, 2, 1),
                         noc_hop=50, unit_static=50)

        self.resource = Resource(
            proc_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(4, 4),
                                   type=NodeRegion.PROC),
            data_regions=(NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(4, 1),
                                     type=NodeRegion.DATA),),
            dim_array=PhyDim2(16, 16), size_gbuf=65536, size_regf=64)

        self.options = Option(partition_hybrid=True, partition_batch=True,
                              partition_ifmaps=True, ntops=10)

        self.ifmap_layouts = {}
        part = PartitionScheme(order=(pe.INPP, pe.BATP, pe.OUTP, pe.OFMP),
                               pdims=((1, 2), (2, 1), (1, 2), (2, 1)))
        for wlkey in self.layers:
            self.ifmap_layouts[wlkey] = partition.get_ofmap_layout(
                self.layers[wlkey].input_layer(), self.batch_size, part,
                self.resource.src_data_region())

    def test_valid_args(self):
        ''' Valid arguments for constructor. '''
        schd = Scheduling(self.layers['BASE'], self.batch_size, self.cost,
                          MapStrategyEyeriss)

        self.assertEqual(schd.layer, self.layers['BASE'])
        self.assertEqual(schd.batch_size, self.batch_size)
        self.assertEqual(schd.cost, self.cost)
        self.assertEqual(schd.map_strategy_class, MapStrategyEyeriss)

    def test_invalid_layer(self):
        ''' Invalid layer argument. '''
        with self.assertRaisesRegexp(TypeError, 'Scheduling: .*layer.*'):
            _ = Scheduling((64, 128, 28, 3), self.batch_size, self.cost,
                           MapStrategyEyeriss)

    def test_invalid_cost(self):
        ''' Invalid cost argument. '''
        with self.assertRaisesRegexp(TypeError, 'Scheduling: .*cost.*'):
            _ = Scheduling(self.layers['BASE'], self.batch_size,
                           tuple(self.cost), MapStrategyEyeriss)

    def test_invalid_map_strategy(self):
        ''' Invalid cost argument. '''
        class _DummyClass(object):  # pylint: disable=too-few-public-methods
            pass

        with self.assertRaisesRegexp(TypeError,
                                     'Scheduling: .*map_strategy_class.*'):
            _ = Scheduling(self.layers['BASE'], self.batch_size, self.cost,
                           _DummyClass)

    def test_schedule_search(self):
        ''' Schedule search. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]
            ifmap_layout = self.ifmap_layouts[wlkey]

            schd = Scheduling(layer, self.batch_size, self.cost,
                              MapStrategyEyeriss)

            condition = SchedulingCondition(resource=self.resource,
                                            ifmap_layout=ifmap_layout)

            res = schd.schedule_search(condition, self.options)

            # Top N.
            self.assertLessEqual(len(res), self.options.ntops)
            self.assertTrue(all(isinstance(r, SchedulingResult) for r in res))
            for idx in range(len(res) - 1):
                self.assertLessEqual(res[idx].total_cost,
                                     res[idx + 1].total_cost)

            # Combination of loop blocking and partitioning.
            for r in res:
                self.assertEqual(r.total_cost,
                                 r.dict_loop['cost'] + r.dict_part['cost'])
                self.assertEqual(r.dict_loop['ops'],
                                 layer.total_ops(self.batch_size))
                self.assertSequenceEqual(r.dict_part['total_nhops'],
                                         [nh * f for nh, f
                                          in zip(r.dict_part['unit_nhops'],
                                                 r.dict_loop['fetch'][0])])
                self.assertEqual(r.dict_part['num_nodes'],
                                 self.resource.proc_region.dim.size())

            # Ofmap layout.
            for r in res:
                self.assertEqual(r.ofmap_layout.frmap.complete_fmap_range()
                                 .size(),
                                 layer.total_ofmap_size(self.batch_size))

    def test_schedule_search_ilayout(self):
        ''' Invalid ifmap_layout. '''
        layer = self.layers['BASE']
        ifmap_layout = self.ifmap_layouts['BASE']

        schd = Scheduling(layer, self.batch_size, self.cost,
                          MapStrategyEyeriss)

        # Shift ifmap out of memory region.
        condition = SchedulingCondition(
            resource=self.resource,
            ifmap_layout=ifmap_layout.view(PhyDim2(1, 1)))

        with self.assertRaisesRegexp(ValueError, 'Scheduling: .*ifmap.*'):
            _ = schd.schedule_search(condition, self.options)

        # Not match layer.
        condition = SchedulingCondition(
            resource=self.resource,
            ifmap_layout=self.ifmap_layouts['POOL'])

        with self.assertRaisesRegexp(ValueError, 'Scheduling: .*ifmap.*'):
            _ = schd.schedule_search(condition, self.options)

    def test_pernode_sched_cache(self):
        ''' Per-node scheduling cache. '''
        layer = self.layers['BASE']
        ifmap_layout = self.ifmap_layouts['BASE']

        schd = Scheduling(layer, self.batch_size, self.cost,
                          MapStrategyEyeriss)

        self.assertEqual(len(schd.pernode_sched_cache), 0)
        self.assertTupleEqual(schd.cache_stats(), (0, 0))

        condition = SchedulingCondition(resource=self.resource,
                                        ifmap_layout=ifmap_layout)

        _ = schd.schedule_search(condition, self.options)

        h, m = schd.cache_stats()
        self.assertEqual(len(schd.pernode_sched_cache), m)
        self.assertEqual(h, 0)
        n = m

        _ = schd.schedule_search(condition, self.options)

        self.assertEqual(len(schd.pernode_sched_cache), n)
        self.assertTupleEqual(schd.cache_stats(), (n, n))

    def test_pernode_sched_cache_key(self):
        ''' Per-node scheduling cache key must be hash-able. '''
        layer = self.layers['BASE']
        ifmap_layout = self.ifmap_layouts['BASE']

        schd = Scheduling(layer, self.batch_size, self.cost,
                          MapStrategyEyeriss)

        condition = SchedulingCondition(resource=self.resource,
                                        ifmap_layout=ifmap_layout)

        _ = schd.schedule_search(condition, self.options)

        h, m = schd.cache_stats()
        self.assertEqual(h, 0)

        # Make another instance.
        rsrc = Resource(**self.resource._asdict())
        opts = Option(**self.options._asdict())
        self.assertNotEqual(id(rsrc), id(self.resource))
        self.assertNotEqual(id(opts), id(self.options))

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 2), (2, 2), (1, 1), (1, 1)))

        _ = schd.schedule_search_per_node(part, rsrc, opts)

        h2, m2 = schd.cache_stats()
        self.assertEqual(h2, h + 1)
        self.assertEqual(m2, m)

