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

from nn_dataflow.core import ConvLayer, LocalRegionLayer, PoolingLayer
from nn_dataflow.core import Cost
from nn_dataflow.core import DataLayout
from nn_dataflow.core import FmapPosition, FmapRange
from nn_dataflow.core import MapStrategyEyeriss
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import Option
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource
from nn_dataflow.core import Scheduling
from nn_dataflow.core import SchedulingCondition, SchedulingResult
from nn_dataflow.core import SchedulingConstraint

class TestScheduling(unittest.TestCase):
    ''' Tests for Scheduling module. '''

    def setUp(self):

        self.layers = {}
        self.layers['BASE'] = ConvLayer(8, 16, 28, 3)
        self.layers['POOL'] = PoolingLayer(16, 28, 2)
        self.layers['LR'] = LocalRegionLayer(16, 28, nreg=3, sreg=1)

        self.batch_size = 4

        self.cost = Cost(mac_op=1, mem_hier=(200, 6, 2, 1),
                         noc_hop=50, idl_unit=50)

        self.none_cstr = SchedulingConstraint()
        self.cstr = SchedulingConstraint(topofm=1, topbat=self.batch_size)

        self.resource = Resource(
            proc_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(4, 4),
                                   type=NodeRegion.PROC),
            dram_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(4, 1),
                                   type=NodeRegion.DRAM),
            src_data_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(4, 1),
                                       type=NodeRegion.DRAM),
            dst_data_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(4, 1),
                                       type=NodeRegion.DRAM),
            dim_array=PhyDim2(16, 16), size_gbuf=65536, size_regf=64,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)

        self.options = Option(partition_hybrid=True, partition_batch=True,
                              partition_ifmaps=True, ntops=10)

        self.ifmap_layouts = {}
        part = PartitionScheme(order=(pe.INPP, pe.BATP, pe.OUTP, pe.OFMP),
                               pdims=((1, 2), (2, 1), (1, 2), (2, 1)))
        for wlkey in self.layers:
            input_layer = self.layers[wlkey].input_layer()
            self.ifmap_layouts[wlkey] = DataLayout(
                frngs=(FmapRange((0, 0, 0, 0),
                                 FmapPosition(b=self.batch_size,
                                              n=input_layer.nofm,
                                              h=input_layer.hofm,
                                              w=input_layer.wofm)),),
                regions=(self.resource.src_data_region,),
                parts=(part.projection(self.resource.src_data_region,
                                       appl2frng=True),))

        self.sched_seq = (2, 0, 1)

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
        with self.assertRaisesRegex(TypeError, 'Scheduling: .*layer.*'):
            _ = Scheduling((64, 128, 28, 3), self.batch_size, self.cost,
                           MapStrategyEyeriss)

    def test_invalid_cost(self):
        ''' Invalid cost argument. '''
        with self.assertRaisesRegex(TypeError, 'Scheduling: .*cost.*'):
            _ = Scheduling(self.layers['BASE'], self.batch_size,
                           tuple(self.cost), MapStrategyEyeriss)

    def test_invalid_map_strategy(self):
        ''' Invalid cost argument. '''
        class _DummyClass():  # pylint: disable=too-few-public-methods
            pass

        with self.assertRaisesRegex(TypeError,
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
                                            constraint=self.cstr,
                                            ifmap_layout=ifmap_layout,
                                            sched_seq=self.sched_seq)

            res = schd.schedule_search(condition, self.options)

            # Top N.
            self.assertLessEqual(len(res), self.options.ntops)
            self.assertTrue(all(isinstance(r, SchedulingResult) for r in res))
            for idx in range(len(res) - 1):
                self.assertLessEqual(res[idx].total_cost,
                                     res[idx + 1].total_cost)

            # Combination of loop blocking and partitioning.
            for r in res:
                self.assertAlmostEqual(r.total_cost,
                                       r.scheme['cost_op']
                                       + r.scheme['cost_access']
                                       + r.scheme['cost_noc']
                                       + r.scheme['cost_static'])
                self.assertEqual(r.total_ops, layer.total_ops(self.batch_size))
                self.assertSequenceEqual(r.scheme['total_nhops'],
                                         [nh * f for nh, f
                                          in zip(r.scheme['unit_nhops'],
                                                 r.scheme['fetch'][0])])
                self.assertEqual(r.num_nodes,
                                 self.resource.proc_region.dim.size())

            # Constraint.
            for r in res:
                self.assertEqual(r.scheme['to'][0], 1)

            # Ofmap layout.
            for r in res:
                self.assertEqual(r.ofmap_layout.complete_fmap_range().size(),
                                 layer.total_ofmap_size(self.batch_size))

            # Sequence number.
            for r in res:
                self.assertTupleEqual(r.sched_seq, condition.sched_seq)

    def test_schedule_search_ilayout(self):
        ''' Invalid ifmap_layout. '''
        layer = self.layers['BASE']

        schd = Scheduling(layer, self.batch_size, self.cost,
                          MapStrategyEyeriss)

        # Shift ifmap out of memory region.
        condition = SchedulingCondition(
            resource=self.resource,
            constraint=self.none_cstr,
            ifmap_layout=self.ifmap_layouts['BASE']._replace(
                regions=tuple(r._replace(origin=PhyDim2(-10, -10))
                              for r in self.ifmap_layouts['BASE'].regions)),
            sched_seq=self.sched_seq)

        with self.assertRaisesRegex(ValueError, 'Scheduling: .*ifmap.*'):
            _ = schd.schedule_search(condition, self.options)

        # Not match layer.
        condition = SchedulingCondition(
            resource=self.resource,
            constraint=self.none_cstr,
            ifmap_layout=self.ifmap_layouts['POOL'],
            sched_seq=self.sched_seq)

        with self.assertRaisesRegex(ValueError, 'Scheduling: .*ifmap.*'):
            _ = schd.schedule_search(condition, self.options)

    def test_schedule_search_nolbs(self):
        ''' Schedule search with no lbs. '''
        layer = self.layers['BASE']
        ifmap_layout = self.ifmap_layouts['BASE']

        schd = Scheduling(layer, self.batch_size, self.cost,
                          MapStrategyEyeriss)

        condition = SchedulingCondition(
            resource=self.resource._replace(size_regf=0),
            constraint=self.none_cstr,
            ifmap_layout=ifmap_layout,
            sched_seq=self.sched_seq)

        res = schd.schedule_search(condition, self.options)

        self.assertFalse(res)

    def test_pernode_sched_cache(self):
        ''' Per-node scheduling cache. '''
        # pylint: disable=no-member
        Scheduling.schedule_search_per_node.cache_clear()

        layer = self.layers['BASE']
        ifmap_layout = self.ifmap_layouts['BASE']

        schd = Scheduling(layer, self.batch_size, self.cost,
                          MapStrategyEyeriss)

        self.assertEqual(schd.schedule_search_per_node.cache_info().currsize, 0)
        self.assertTupleEqual(schd.cache_stats(), (0, 0))

        condition = SchedulingCondition(resource=self.resource,
                                        constraint=self.cstr,
                                        ifmap_layout=ifmap_layout,
                                        sched_seq=self.sched_seq)

        Scheduling.schedule_search.cache_clear()
        _ = schd.schedule_search(condition, self.options)

        h, m = schd.cache_stats()
        self.assertEqual(schd.schedule_search_per_node.cache_info().currsize, m)
        self.assertEqual(h, 0)
        n = m

        Scheduling.schedule_search.cache_clear()
        _ = schd.schedule_search(condition, self.options)

        self.assertEqual(schd.schedule_search_per_node.cache_info().currsize, n)
        self.assertTupleEqual(schd.cache_stats(), (n, n))

    def test_pernode_sched_cache_key(self):
        ''' Per-node scheduling cache key must be hash-able. '''
        # pylint: disable=no-member
        Scheduling.schedule_search.cache_clear()
        Scheduling.schedule_search_per_node.cache_clear()

        layer = self.layers['BASE']
        ifmap_layout = self.ifmap_layouts['BASE']

        schd = Scheduling(layer, self.batch_size, self.cost,
                          MapStrategyEyeriss)

        condition = SchedulingCondition(resource=self.resource,
                                        constraint=self.cstr,
                                        ifmap_layout=ifmap_layout,
                                        sched_seq=self.sched_seq)

        _ = schd.schedule_search(condition, self.options)

        h, m = schd.cache_stats()
        self.assertEqual(h, 0)

        # Make another instance.
        rsrc = Resource(**self.resource._asdict())
        cstr = self.cstr
        opts = Option(**self.options._asdict())
        self.assertNotEqual(id(rsrc), id(self.resource))
        self.assertNotEqual(id(opts), id(self.options))

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 4), (2, 1), (1, 1), (1, 1)))

        _ = schd.schedule_search_per_node(part, rsrc, cstr, opts)

        h2, m2 = schd.cache_stats()
        self.assertEqual(h2, h + 1)
        self.assertEqual(m2, m)

