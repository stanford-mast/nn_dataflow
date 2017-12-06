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

from nn_dataflow.core import DataLayout
from nn_dataflow.core import FmapRange, FmapRangeMap
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import SchedulingResult

class TestSchedulingResult(unittest.TestCase):
    ''' Tests for SchedulingResult. '''

    def setUp(self):

        self.dict_loop = OrderedDict([('cost', 1.234),
                                      ('time', 123.4),
                                      ('proc_time', 59),
                                      ('bus_time', 40),
                                      ('dram_time', 120),
                                      ('ops', 1234),
                                      ('access', [[2, 3, 4],
                                                  [30, 40, 50],
                                                  [400, 500, 600],
                                                  [5000, 6000, 7000]]),
                                      ('remote_gbuf_access', [0, 0, 0]),
                                     ])
        self.dict_part = OrderedDict([('cost', 9.876),
                                      ('total_nhops', [123, 456, 789]),
                                     ])

        frmap = FmapRangeMap()
        frmap.add(FmapRange((0, 0, 0, 0), (2, 4, 16, 16)), (PhyDim2(0, 0),))
        self.ofmap_layout = DataLayout(origin=PhyDim2(0, 0), frmap=frmap,
                                       type=NodeRegion.DATA)

        self.sched_seq = (2, 0, 0)

    def test_valid_args(self):
        ''' Valid arguments. '''
        result = SchedulingResult(dict_loop=self.dict_loop,
                                  dict_part=self.dict_part,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertIn('ops', result.dict_loop)
        self.assertIn('total_nhops', result.dict_part)
        self.assertEqual(result.ofmap_layout, self.ofmap_layout)
        self.assertTupleEqual(result.sched_seq, self.sched_seq)

    def test_invalid_dict_loop(self):
        ''' Invalid dict_loop. '''
        with self.assertRaisesRegexp(TypeError,
                                     'SchedulingResult: .*dict_loop.*'):
            _ = SchedulingResult(dict_loop={},
                                 dict_part=self.dict_part,
                                 ofmap_layout=self.ofmap_layout,
                                 sched_seq=self.sched_seq)

    def test_invalid_dict_part(self):
        ''' Invalid dict_part. '''
        with self.assertRaisesRegexp(TypeError,
                                     'SchedulingResult: .*dict_part.*'):
            _ = SchedulingResult(dict_loop=self.dict_loop,
                                 dict_part={},
                                 ofmap_layout=self.ofmap_layout,
                                 sched_seq=self.sched_seq)

    def test_invalid_ofmap_layout(self):
        ''' Invalid ofmap_layout. '''
        with self.assertRaisesRegexp(TypeError,
                                     'SchedulingResult: .*ofmap_layout.*'):
            _ = SchedulingResult(dict_loop=self.dict_loop,
                                 dict_part=self.dict_part,
                                 ofmap_layout=None,
                                 sched_seq=self.sched_seq)

    def test_invalid_sched_seq(self):
        ''' Invalid sched_seq. '''
        with self.assertRaisesRegexp(TypeError,
                                     'SchedulingResult: .*sched_seq.*'):
            _ = SchedulingResult(dict_loop=self.dict_loop,
                                 dict_part=self.dict_part,
                                 ofmap_layout=self.ofmap_layout,
                                 sched_seq=list(self.sched_seq))

        with self.assertRaisesRegexp(ValueError,
                                     'SchedulingResult: .*sched_seq.*'):
            _ = SchedulingResult(dict_loop=self.dict_loop,
                                 dict_part=self.dict_part,
                                 ofmap_layout=self.ofmap_layout,
                                 sched_seq=self.sched_seq[:-1])

    def test_total_cost(self):
        ''' Accessor total_cost. '''
        result = SchedulingResult(dict_loop=self.dict_loop,
                                  dict_part=self.dict_part,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_cost, 1.234 + 9.876)

    def test_total_time(self):
        ''' Accessor total_time. '''
        result = SchedulingResult(dict_loop=self.dict_loop,
                                  dict_part=self.dict_part,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_time, 123.4)

        self.assertGreaterEqual(result.total_time, result.total_node_time)
        self.assertGreaterEqual(result.total_time, result.total_dram_time)

    def test_total_node_time(self):
        ''' Accessor total_node_time. '''
        result = SchedulingResult(dict_loop=self.dict_loop,
                                  dict_part=self.dict_part,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_node_time, 59 + 40)

    def test_total_dram_time(self):
        ''' Accessor total_dram_time. '''
        result = SchedulingResult(dict_loop=self.dict_loop,
                                  dict_part=self.dict_part,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertAlmostEqual(result.total_dram_time, 120)

    def test_total_ops(self):
        ''' Accessor total_ops. '''
        result = SchedulingResult(dict_loop=self.dict_loop,
                                  dict_part=self.dict_part,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertEqual(result.total_ops, 1234)

    def test_total_accesses(self):
        ''' Accessor total_accesses. '''
        result = SchedulingResult(dict_loop=self.dict_loop,
                                  dict_part=self.dict_part,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertSequenceEqual(result.total_accesses,
                                 [9, 120, 1500, 18000])

    def test_total_accesses_rgbuf(self):
        ''' Accessor total_accesses remote gbuf. '''
        dict_loop = self.dict_loop.copy()
        dict_loop['remote_gbuf_access'] = [10, 20, 30]
        result = SchedulingResult(dict_loop=dict_loop,
                                  dict_part=self.dict_part,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertSequenceEqual(result.total_accesses,
                                 [9, 120 + 60, 1500, 18000])

    def test_total_noc_hops(self):
        ''' Accessor total_noc_hops. '''
        result = SchedulingResult(dict_loop=self.dict_loop,
                                  dict_part=self.dict_part,
                                  ofmap_layout=self.ofmap_layout,
                                  sched_seq=self.sched_seq)
        self.assertEqual(result.total_noc_hops, 1368)

