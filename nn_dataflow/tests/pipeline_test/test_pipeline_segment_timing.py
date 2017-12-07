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

from collections import OrderedDict

from nn_dataflow.core import InputLayer, FCLayer, PoolingLayer
from nn_dataflow.core import DataLayout
from nn_dataflow.core import FmapRange, FmapRangeMap
from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import Network
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import PipelineSegmentTiming
from nn_dataflow.core import SchedulingResult

from . import TestPipelineFixture

class TestPipelineSegmentTiming(TestPipelineFixture):
    ''' Tests for PipelineSegmentTiming. '''

    def setUp(self):
        super(TestPipelineSegmentTiming, self).setUp()

        self.net1 = self.net['net1']

        self.net4 = self.net['net4']

        self.netlr = Network('net1')
        self.netlr.set_input(InputLayer(10, 1))
        self.netlr.add('0p1', PoolingLayer(10, 1, 1))
        self.netlr.add('0p2', PoolingLayer(10, 1, 1))
        self.netlr.add('0p3', PoolingLayer(10, 1, 1))
        self.netlr.add('1', FCLayer(10, 20))

        self.dict_part = OrderedDict([('cost', 9.876)])

        frmap = FmapRangeMap()
        frmap.add(FmapRange((0, 0, 0, 0), (2, 4, 16, 16)), (PhyDim2(0, 0),))
        self.ofmap_layout = DataLayout(origin=PhyDim2(0, 0), frmap=frmap,
                                       type=NodeRegion.DATA)

    def test_valid_args(self):
        ''' Valid arguments. '''
        timing = PipelineSegmentTiming(self.net1, 3)
        self.assertIs(timing.network, self.net1)
        self.assertEqual(timing.seg_idx, 3)

    def test_invalid_network(self):
        ''' Invalid network. '''
        with self.assertRaisesRegexp(TypeError,
                                     'PipelineSegmentTiming: .*network.*'):
            _ = PipelineSegmentTiming(self.net1.input_layer(), 3)

    def test_add(self):
        ''' add(). '''
        timing = PipelineSegmentTiming(self.net1, 3)

        timing.add('0', self._make_sched_res((3, 0, 0), 123,
                                             top_to=3, top_tb=4))
        self.assertTupleEqual(timing.last_sched_seq, (3, 0, 0))
        self.assertFalse(timing.timing_list[-1][-1].fused)

        timing.add('1', self._make_sched_res((3, 1, 0), 141,
                                             top_ti=3, top_tb=2))
        self.assertTupleEqual(timing.last_sched_seq, (3, 1, 0))
        self.assertFalse(timing.timing_list[-1][-1].fused)

        timing.add('1p', self._make_sched_res((3, 1, 1), 12,
                                              top_ti=3, top_tb=8))
        self.assertTupleEqual(timing.last_sched_seq, (3, 1, 1))
        self.assertTrue(timing.timing_list[-1][-1].fused)

        self.assertEqual(timing.bat_ngrp, 2)
        self.assertEqual(len(timing.timing_list), 2)
        self.assertEqual(len(timing.timing_list[0]), 1)
        self.assertEqual(len(timing.timing_list[1]), 2)

    def test_add_all_lr(self):
        ''' add() all LocalRegionLayer. '''
        timing = PipelineSegmentTiming(self.netlr, 2)

        timing.add('0p1', self._make_sched_res((2, 0, 0), 40, top_tb=4))
        self.assertFalse(timing.timing_list[-1][-1].fused)
        timing.add('0p2', self._make_sched_res((2, 0, 1), 80, top_tb=4))
        self.assertTrue(timing.timing_list[-1][-1].fused)
        timing.add('0p3', self._make_sched_res((2, 0, 2), 60, top_tb=2))
        self.assertTrue(timing.timing_list[-1][-1].fused)
        timing.add('1', self._make_sched_res((2, 1, 0), 800, top_tb=2))
        self.assertFalse(timing.timing_list[-1][-1].fused)

    def test_add_invalid_sched_seq(self):
        ''' add(), invalid sched seq. '''
        timing = PipelineSegmentTiming(self.net1, 3)
        timing.add('0', self._make_sched_res((3, 0, 0), 123))

        with self.assertRaisesRegexp(ValueError,
                                     'PipelineSegmentTiming: .*belong to.*'):
            timing.add('1', self._make_sched_res((2, 1, 0), 123))

        with self.assertRaisesRegexp(ValueError,
                                     'PipelineSegmentTiming: .*follow.*'):
            timing.add('1p', self._make_sched_res((3, 1, 1), 123))

    def test_add_already_in(self):
        ''' add(), layer already in. '''
        timing = PipelineSegmentTiming(self.net1, 3)
        timing.add('0', self._make_sched_res((3, 0, 0), 123))
        with self.assertRaisesRegexp(ValueError,
                                     'PipelineSegmentTiming: .*layer 0.*'):
            timing.add('0', self._make_sched_res((3, 1, 0), 123))

    def test_time_bat_ngrp(self):
        ''' time() and critical_time() bat_ngrp. '''
        timing = PipelineSegmentTiming(self.net1, 3)
        timing.add('0', self._make_sched_res((3, 0, 0), 120, top_tb=4))
        timing.add('1', self._make_sched_res((3, 1, 0), 130, top_tb=8))
        timing.add('1p', self._make_sched_res((3, 1, 1), 20, top_tb=2))
        timing.add('2', self._make_sched_res((3, 2, 0), 138, top_tb=2))
        self.assertEqual(timing.critical_time(), 150)
        self.assertEqual(timing.time(), 120 // 2 + 130 + 20 + 138 // 2)

    def test_time_ifm_ofm_ngrp(self):
        ''' time() and critical_time() ifm_ngrp and ofm_ngrp. '''

        # Single-group wait, first critical.

        timing = PipelineSegmentTiming(self.net1, 3)
        timing.add('0', self._make_sched_res((3, 0, 0), 120,
                                             top_to=3, top_tb=2))
        timing.add('1', self._make_sched_res((3, 1, 0), 90,
                                             top_ti=3, top_tb=2))
        self.assertEqual(timing.critical_time(), 120)
        # Layer 0 is critical. Layer 0 last BAT group starts at 120 - 120 // 2.
        # Layer 1 last BAT group starts 120 // 2 // 3 later, which takes 90 //
        # 2.
        self.assertEqual(timing.time(),
                         120 - 120 // 2 + 120 // 2 // 3 + 90 // 2)

        # Single-group wait, second critical.

        timing = PipelineSegmentTiming(self.net1, 3)
        timing.add('0', self._make_sched_res((3, 0, 0), 120,
                                             top_to=3, top_tb=2))
        timing.add('1', self._make_sched_res((3, 1, 0), 150,
                                             top_ti=3, top_tb=2))
        self.assertEqual(timing.critical_time(), 150)
        # Layer 1 is critical. Layer 1 first BAT group starts at 120 // 2 // 3,
        # and takes 150 for all its BAT groups.
        self.assertEqual(timing.time(), 120 // 2 // 3 + 150)

        # All-group wait, first critical.

        timing = PipelineSegmentTiming(self.net1, 3)
        timing.add('0', self._make_sched_res((3, 0, 0), 120,
                                             top_to=3, top_tb=2))
        timing.add('1', self._make_sched_res((3, 1, 0), 90,
                                             top_to=3, top_tb=2))
        self.assertEqual(timing.critical_time(), 120)
        self.assertEqual(timing.time(), 120 + 90 // 2)

        # All-group wait, second critical.

        timing = PipelineSegmentTiming(self.net1, 3)
        timing.add('0', self._make_sched_res((3, 0, 0), 120,
                                             top_ti=3, top_tb=2))
        timing.add('1', self._make_sched_res((3, 1, 0), 150,
                                             top_ti=3, top_tb=2))
        self.assertEqual(timing.critical_time(), 150)
        self.assertEqual(timing.time(), 120 // 2 + 150)

    def test_time_linear(self):
        ''' time() and critical_time() linear. '''
        timing = PipelineSegmentTiming(self.net1, 3)
        timing.add('0', self._make_sched_res((3, 0, 0), 120,
                                             top_ti=3, top_tb=4))
        timing.add('1', self._make_sched_res((3, 1, 0), 130,
                                             top_to=3, top_tb=8))
        timing.add('1p', self._make_sched_res((3, 1, 1), 20,
                                              top_to=3, top_tb=2))
        timing.add('2', self._make_sched_res((3, 2, 0), 138,
                                             top_ti=3, top_tb=2))
        self.assertEqual(timing.critical_time(), 150)
        # Layer 1 is critical. Layer 1+1p first BAT group starts at 120 // 2,
        # and last BAT group starts at 150 // 2 later. Layer 2 last BAT group
        # starts 150 // 2 // 3 later, and takes 138 // 2.
        self.assertEqual(timing.time(),
                         120 // 2 + 150 // 2 + 150 // 2 // 3 + 138 // 2)

    def test_time_branch(self):
        ''' time() and critical_time() branch. '''

        # Single-group wait.

        timing = PipelineSegmentTiming(self.net4, 3)
        timing.add('6', self._make_sched_res((3, 0, 0), 120,
                                             top_ti=3, top_tb=4))
        timing.add('7', self._make_sched_res((3, 1, 0), 150,
                                             top_to=3, top_tb=8))
        timing.add('8', self._make_sched_res((3, 2, 0), 144,
                                             top_ti=3, top_tb=2))
        timing.add('9', self._make_sched_res((3, 3, 0), 168,
                                             top_ti=3, top_tb=4))
        self.assertEqual(timing.critical_time(), 168)
        # Layer 9 is critical. Layer 7 first BAT group starts at 120 // 2.
        # Layer 8 and 9 first BAT group starts at 150 // 2 // 3 later, and all
        # groups of layer 9 take 168.
        self.assertEqual(timing.time(),
                         120 // 2 + 150 // 2 // 3 + 168)

        # All-group wait.

        timing = PipelineSegmentTiming(self.net4, 3)
        timing.add('6', self._make_sched_res((3, 0, 0), 120, top_tb=4))
        timing.add('7', self._make_sched_res((3, 1, 0), 150, top_tb=8))
        timing.add('8', self._make_sched_res((3, 2, 0), 144, top_tb=2))
        timing.add('9', self._make_sched_res((3, 3, 0), 132, top_tb=4))
        self.assertEqual(timing.critical_time(), 150)
        # Layer 7 is critical. Layer 7 first BAT group starts at 120 // 2, and
        # layer 7 last BAT group ends at 150 later, at which time layer 8 and 9
        # last BAT group starts, and takes 144 // 2.
        self.assertEqual(timing.time(), 120 // 2 + 150 + 144 // 2)

    def test_time_all_lr(self):
        ''' time() and critical_time() all LocalRegionLayer. '''
        timing = PipelineSegmentTiming(self.netlr, 2)
        timing.add('0p1', self._make_sched_res((2, 0, 0), 40,
                                               top_to=10, top_tb=4))
        timing.add('0p2', self._make_sched_res((2, 0, 1), 80,
                                               top_to=10, top_tb=4))
        timing.add('0p3', self._make_sched_res((2, 0, 2), 60,
                                               top_to=10, top_tb=2))
        timing.add('1', self._make_sched_res((2, 1, 0), 800,
                                             top_ti=5, top_tb=2))
        self.assertEqual(timing.critical_time(), 800)
        # Layer 1 is critical. Layer 1 first BAT group starts at (40 + 80 + 60)
        # // 2 // 5, and takes 800.
        self.assertEqual(timing.time(), (40 + 80 + 60) // 2 // 5 + 800)

    def test_time_single_spatial(self):
        ''' time() and critical_time() for single-spatial segment. '''

        for net_name in self.net:
            if not net_name.startswith('net'):
                continue
            net = self.net[net_name]

            for seg in self._gen_all_segment(net, temporal=True):
                if not seg.valid:
                    continue
                self.assertEqual(len(seg), 1)

                timing = PipelineSegmentTiming(net, 0)
                for idx, layer in enumerate(seg[0]):
                    timing.add(layer,
                               self._make_sched_res((0, 0, idx),
                                                    (40 + idx * 7 % 3) * 16,
                                                    top_to=4, top_ti=4,
                                                    top_tb=4))

                self.assertEqual(timing.critical_time(), timing.time())

    def test_time_dram_time(self):
        ''' time() and critical_time() dominated by DRAM time. '''
        timing = PipelineSegmentTiming(self.net1, 3)
        timing.add('0', self._make_sched_res((3, 0, 0), 120, dram_time=100,
                                             top_ti=3, top_tb=4))
        timing.add('1', self._make_sched_res((3, 1, 0), 130, dram_time=140,
                                             top_to=3, top_tb=8))
        timing.add('1p', self._make_sched_res((3, 1, 1), 20, dram_time=10,
                                              top_to=3, top_tb=2))
        timing.add('2', self._make_sched_res((3, 2, 0), 138, dram_time=100,
                                             top_ti=3, top_tb=2))
        self.assertEqual(timing.critical_time(), 160)
        self.assertEqual(timing.time(), 100 + 140 + 10 + 100)
        self.assertEqual(timing.dram_time(), timing.time())
        self.assertLess(timing.node_time(), timing.time())

    def _make_sched_res(self, sched_seq, time, top_ti=1, top_to=1, top_tb=1,
                        top_ord=range(le.NUM), dram_time=0):
        dict_loop = OrderedDict()
        dict_loop['cost'] = 1.234
        dict_loop['time'] = max(time, dram_time)
        dict_loop['proc_time'] = time
        dict_loop['bus_time'] = 0
        dict_loop['dram_time'] = dram_time
        dict_loop['ti'] = [top_ti, 1, 1]
        dict_loop['to'] = [top_to, 1, 1]
        dict_loop['tb'] = [top_tb, 1, 1]
        dict_loop['orders'] = [top_ord, range(le.NUM), range(le.NUM)]
        return SchedulingResult(dict_loop=dict_loop,
                                dict_part=self.dict_part,
                                ofmap_layout=self.ofmap_layout,
                                sched_seq=sched_seq)

