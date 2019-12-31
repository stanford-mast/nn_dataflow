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
from nn_dataflow.core import FmapRange
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import SchedulingResult

class TestSchedulingResult(unittest.TestCase):
    ''' Tests for SchedulingResult. '''

    def setUp(self):

        self.scheme = OrderedDict([('cost', 9.876 + 1.234),
                                   ('time', 123.4),
                                   ('ops', 1234),
                                   ('num_nodes', 4),
                                   ('cost_op', 1.234),
                                   ('cost_access', 9.876),
                                   ('cost_noc', 0),
                                   ('cost_static', 0),
                                   ('proc_time', 59),
                                   ('bus_time', 40),
                                   ('dram_time', 120),
                                   ('access', [[2, 3, 4],
                                               [30, 40, 50],
                                               [400, 500, 600],
                                               [5000, 6000, 7000]]),
                                   ('remote_gbuf_access', [0, 0, 0]),
                                   ('total_nhops', [123, 456, 789]),
                                   ('fetch', [[1, 2, 1], [3, 4, 5]]),
                                  ])

        part = PartitionScheme(order=range(pe.NUM), pdims=[(1, 1)] * pe.NUM)
        self.ofmap_layout = DataLayout(
            frngs=(FmapRange((0, 0, 0, 0), (2, 4, 16, 16)),),
            regions=(NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                type=NodeRegion.DRAM),),
            parts=(part,))

        self.sched_seq = (2, 0, 0)

    def test_valid_args(self):
        ''' Valid arguments. '''
        result = SchedulingResult(scheme=self.scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertIn('ops', result.scheme)
        self.assertIn('total_nhops', result.scheme)
        self.assertEqual(result.ofmap_layout, self.ofmap_layout)
        self.assertTupleEqual(result.sched_seq, self.sched_seq)

    def test_invalid_scheme(self):
        ''' Invalid scheme. '''
        with self.assertRaisesRegex(TypeError,
                                    'SchedulingResult: .*scheme.*'):
            _ = SchedulingResult(scheme={},
                                 ofmap_layout=self.ofmap_layout,
                                 sched_seq=self.sched_seq)

    def test_invalid_ofmap_layout(self):
        ''' Invalid ofmap_layout. '''
        with self.assertRaisesRegex(TypeError,
                                    'SchedulingResult: .*ofmap_layout.*'):
            _ = SchedulingResult(scheme=self.scheme,
                                 ofmap_layout=None,
                                 sched_seq=self.sched_seq)

    def test_invalid_sched_seq(self):
        ''' Invalid sched_seq. '''
        with self.assertRaisesRegex(TypeError,
                                    'SchedulingResult: .*sched_seq.*'):
            _ = SchedulingResult(scheme=self.scheme,
                                 ofmap_layout=self.ofmap_layout,
                                 sched_seq=list(self.sched_seq))

        with self.assertRaisesRegex(ValueError,
                                    'SchedulingResult: .*sched_seq.*'):
            _ = SchedulingResult(scheme=self.scheme,
                                 ofmap_layout=self.ofmap_layout,
                                 sched_seq=self.sched_seq[:-1])

    def test_total_cost(self):
        ''' Accessor total_cost. '''
        result = SchedulingResult(scheme=self.scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_cost, 1.234 + 9.876)

    def test_total_time(self):
        ''' Accessor total_time. '''
        result = SchedulingResult(scheme=self.scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_time, 123.4)

        self.assertGreaterEqual(result.total_time, result.total_node_time)
        self.assertGreaterEqual(result.total_time, result.total_dram_time)

    def test_total_node_time(self):
        ''' Accessor total_node_time. '''
        result = SchedulingResult(scheme=self.scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_node_time, max(59, 40))

        scheme = self.scheme
        scheme['bus_time'] = 100
        result = SchedulingResult(scheme=scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_node_time, max(59, 100))

    def test_total_dram_time(self):
        ''' Accessor total_dram_time. '''
        result = SchedulingResult(scheme=self.scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_dram_time, 120)

    def test_total_proc_time(self):
        ''' Accessor total_proc_time. '''
        result = SchedulingResult(scheme=self.scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_proc_time, 59)

        scheme = self.scheme
        scheme['bus_time'] = 100
        result = SchedulingResult(scheme=scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_proc_time, 59)

    def test_total_ops(self):
        ''' Accessor total_ops. '''
        result = SchedulingResult(scheme=self.scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertEqual(result.total_ops, 1234)

    def test_total_accesses(self):
        ''' Accessor total_cost. '''
        result = SchedulingResult(scheme=self.scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertSequenceEqual(result.total_accesses,
                                 [9, 120, 1500, 18000])

    def test_total_accesses_rgbuf(self):
        ''' Accessor total_accesses remote gbuf. '''
        scheme = self.scheme.copy()
        scheme['remote_gbuf_access'] = [10, 20, 30]
        result = SchedulingResult(scheme=scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertSequenceEqual(result.total_accesses,
                                 [9, 120 + 60, 1500, 18000])

    def test_total_noc_hops(self):
        ''' Accessor total_noc_hops. '''
        result = SchedulingResult(scheme=self.scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertEqual(result.total_noc_hops, 1368)

    def test_num_nodes(self):
        ''' Accessor num_nodes. '''
        result = SchedulingResult(scheme=self.scheme,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertEqual(result.num_nodes, 4)

