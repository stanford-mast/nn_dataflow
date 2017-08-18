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
import sys
import StringIO

from nn_dataflow.core import Cost
from nn_dataflow.core import MapStrategy, MapStrategyEyeriss
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import NNDataflow
from nn_dataflow.core import Option
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource

from nn_dataflow.nns import import_network

class TestNNDataflow(unittest.TestCase):
    ''' Tests for NNDataflow module. '''

    def setUp(self):

        self.alex_net = import_network('alex_net')
        self.vgg_net = import_network('vgg_net')

        self.map_strategy = MapStrategyEyeriss

        self.resource = Resource(proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                                        dim=PhyDim2(1, 1),
                                                        type=NodeRegion.PROC),
                                 data_regions=(NodeRegion(origin=PhyDim2(0, 0),
                                                          dim=PhyDim2(1, 1),
                                                          type=NodeRegion.DATA),
                                              ),
                                 dim_array=PhyDim2(16, 16),
                                 size_gbuf=128 * 1024 // 2,  # 128 kB
                                 size_regf=512 // 2,  # 512 B
                                )

        self.cost = Cost(mac_op=1,
                         mem_hier=(200, 6, 2, 1),
                         noc_hop=0,
                         unit_static=0)

        self.options = Option()

    def test_invalid_network(self):
        ''' Invalid network argument. '''
        with self.assertRaisesRegexp(TypeError, 'NNDataflow: .*network.*'):
            _ = NNDataflow(self.alex_net.input_layer(), 4,
                           self.resource, self.cost, self.map_strategy)

    def test_invalid_resource(self):
        ''' Invalid network argument. '''
        with self.assertRaisesRegexp(TypeError, 'NNDataflow: .*resource.*'):
            _ = NNDataflow(self.alex_net, 4,
                           self.resource.proc_region, self.cost,
                           self.map_strategy)

    def test_invalid_cost(self):
        ''' Invalid network argument. '''
        with self.assertRaisesRegexp(TypeError, 'NNDataflow: .*cost.*'):
            _ = NNDataflow(self.alex_net, 4,
                           self.resource, self.cost._asdict(),
                           self.map_strategy)

    def test_invalid_map_strategy(self):
        ''' Invalid map_strategy argument. '''
        class _DummyClass(object):  # pylint: disable=too-few-public-methods
            pass

        with self.assertRaisesRegexp(TypeError, 'NNDataflow: .*map_strategy.*'):
            _ = NNDataflow(self.alex_net, 4,
                           self.resource, self.cost, _DummyClass)

    def test_verbose(self):
        ''' Verbose mode. '''
        network = self.alex_net

        batch_size = 16

        options = Option(sw_gbuf_bypass=(True, True, True),
                         sw_solve_loopblocking=True,
                         verbose=True)

        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         self.map_strategy)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout = StringIO.StringIO()
        sys.stderr = stderr = StringIO.StringIO()

        tops, _ = nnd.schedule_search(options)

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        stdout_value = stdout.getvalue()
        stderr_value = stderr.getvalue()
        stdout.close()
        stderr.close()

        self.assertTrue(tops)

        self.assertFalse(stdout_value)
        for layer in network:
            self.assertIn(layer, stderr_value)

    def test_no_valid_dataflow(self):
        ''' No valid dataflow is found. '''

        # Very small REGF.
        self.resource = Resource(proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                                        dim=PhyDim2(1, 1),
                                                        type=NodeRegion.PROC),
                                 data_regions=(NodeRegion(origin=PhyDim2(0, 0),
                                                          dim=PhyDim2(1, 1),
                                                          type=NodeRegion.DATA),
                                              ),
                                 dim_array=PhyDim2(16, 16),
                                 size_gbuf=128 * 1024 // 2,  # 128 kB
                                 size_regf=2,
                                )

        nnd = NNDataflow(self.alex_net, 4, self.resource, self.cost,
                         self.map_strategy)
        tops, _ = nnd.schedule_search(self.options)

        self.assertFalse(tops)

    def test_scheduling_failure(self):
        ''' Layer scheduling failure. '''
        network = self.alex_net

        batch_size = 16

        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         MapStrategy)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout = StringIO.StringIO()
        sys.stderr = stderr = StringIO.StringIO()

        with self.assertRaises(NotImplementedError):
            _ = nnd.schedule_search(self.options)

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        stdout_value = stdout.getvalue()
        stderr_value = stderr.getvalue()
        stdout.close()
        stderr.close()

        self.assertFalse(stdout_value)
        self.assertIn('Failed', stderr_value)

    def test_eyeriss_isca16(self):
        '''
        Reproduce Eyeriss ISCA'16 paper Fig. 10.
        '''
        network = self.alex_net

        batch_size = 16

        nnd = NNDataflow(network, batch_size, self.resource, self.cost,
                         self.map_strategy)
        tops, _ = nnd.schedule_search(self.options)
        self.assertTrue(tops)
        dfsch = tops[0]

        ## Check results.

        # Results as cost for each component:
        header = 'ALU, DRAM, Buffer, Array, RF'
        cost_bkdn = {}

        for layer in ['conv{}'.format(i) for i in range(1, 6)] \
                + ['fc{}'.format(i) for i in range(1, 4)]:
            op_cost = 0
            access_cost = [0] * me.NUM

            for layer_part in network:
                if not layer_part or not layer_part.startswith(layer):
                    continue
                sr = dfsch[layer_part]
                op_cost += sr.total_ops * self.cost.mac_op
                access_cost = [ac + a * c for ac, a, c
                               in zip(access_cost, sr.total_accesses,
                                      self.cost.mem_hier)]

            cost_bkdn[layer] = []
            # To 1e9.
            cost_bkdn[layer].append(op_cost / 1e9)
            cost_bkdn[layer].append(access_cost[me.DRAM] / 1e9)
            cost_bkdn[layer].append(access_cost[me.GBUF] / 1e9)
            cost_bkdn[layer].append(access_cost[me.ITCN] / 1e9)
            cost_bkdn[layer].append(access_cost[me.REGF] / 1e9)

        # Check the major parts: ALU, DRAM, RF.
        major_cost_bkdn_ref = {'conv1': [1.69, 2.46, 6.75],
                               'conv2': [3.58, 2.27, 14.33],
                               'conv3': [2.39, 2.02, 9.57],
                               'conv4': [1.79, 1.57, 7.18],
                               'conv5': [1.20, 1.05, 4.78],
                               'fc1':   [0.60, 7.78, 2.42],
                               'fc2':   [0.27, 3.39, 1.07],
                               'fc3':   [0.07, 0.84, 0.26],
                              }
        for layer in cost_bkdn:
            success = all(abs(a - b) < 0.1 for a, b
                          in zip(cost_bkdn[layer][:2] + cost_bkdn[layer][-1:],
                                 major_cost_bkdn_ref[layer]))
            self.assertTrue(success,
                            'test_eyeriss_isca16: '
                            'ALU, DRAM, RF cost diff in layer {}.\n'
                            'header: {}\n'
                            'actual: {}\nref: {}'
                            .format(layer, header, cost_bkdn[layer],
                                    major_cost_bkdn_ref[layer]))

    def test_eyeriss_isscc16(self):
        '''
        Reproduce Eyeriss ISSCC'16 paper Fig. 14.5.6, JSSC'17 paper Table V.
        '''
        network = self.alex_net

        batch_size = 4

        resource = Resource(proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                                   dim=PhyDim2(1, 1),
                                                   type=NodeRegion.PROC),
                            data_regions=(NodeRegion(origin=PhyDim2(0, 0),
                                                     dim=PhyDim2(1, 1),
                                                     type=NodeRegion.DATA),),
                            dim_array=PhyDim2(12, 14),
                            size_gbuf=108 * 1024 // 2,  # 108 kB
                            size_regf=261,  # 225 + 12 + 24
                           )

        cost = Cost(mac_op=2e-12,
                    mem_hier=(460e-12, 15e-12, 4e-12, 1e-12),  # pJ/16-b
                    noc_hop=0,
                    unit_static=30e-3 / 200e6)  # 30 mW GBUF + REGF

        nnd = NNDataflow(network, batch_size, resource, cost,
                         self.map_strategy)
        tops, _ = nnd.schedule_search(self.options)
        self.assertTrue(tops)
        dfsch = tops[0]

        ## Check results.

        # Results as stats of the rows in the table.
        header = 'Power, Processing Latency, Ops, Active PEs, Filter size'
        stats = {}

        for layer in ['conv{}'.format(i) for i in range(1, 6)]:
            onchip_cost = 0
            time = 0
            ops = 0
            fil_size = 0

            for layer_part in network:
                if not layer_part or not layer_part.startswith(layer):
                    continue
                sr = dfsch[layer_part]
                onchip_cost += sr.total_cost \
                        - sr.total_accesses[me.DRAM] * cost.mem_hier[me.DRAM]
                time += sr.total_time
                ops += sr.total_ops
                fil_size += network[layer_part].total_filter_size()

            power = onchip_cost / (time / 200e6) * 1e3  # mW
            active_pes = int(ops / time)

            stats[layer] = []
            stats[layer].append(power)
            stats[layer].append(time / 200.e3)  # cycles to ms
            stats[layer].append(ops / 1e6)  # to MOPs
            stats[layer].append(active_pes)
            stats[layer].append(fil_size / 1e3)  # to k

        # Check.
        stats_ref = {'conv1': [332, 16.5, 421.66, 151, 34.8],  # Act PE 154
                     'conv2': [288, 39.2, 895.79, 135, 307.2],
                     'conv3': [266, 21.8, 598.1, 156, 884.7],
                     'conv4': [235, 16.0, 448.6, 156, 663.6],
                     'conv5': [236, 10.0, 299.0, 156, 442.4],
                    }
        for layer in stats:
            success = (0.6 * stats_ref[layer][0]
                       < stats[layer][0]
                       < stats_ref[layer][0]) \
                    and (0.8 * stats_ref[layer][1]
                         < stats[layer][1]
                         < stats_ref[layer][1]) \
                    and all(abs(a - b) < 0.1 for a, b
                            in zip(stats[layer][2:], stats_ref[layer][2:]))
            self.assertTrue(success,
                            'test_eyeriss_isscc16: '
                            'stats diff in layer {}.\n'
                            'header: {}\n'
                            'actual: {}\nref: {}'
                            .format(layer, header, stats[layer],
                                    stats_ref[layer]))

    def test_eyeriss_asplos17(self):
        '''
        Reproduce TETRIS ASPLOS'17 paper Figure 8.
        '''
        network = self.alex_net

        batch_size = 16

        ## L-1 configuration.

        resource = Resource(proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                                   dim=PhyDim2(1, 1),
                                                   type=NodeRegion.PROC),
                            data_regions=(NodeRegion(origin=PhyDim2(0, 0),
                                                     dim=PhyDim2(1, 1),
                                                     type=NodeRegion.DATA),),
                            dim_array=PhyDim2(16, 16),
                            size_gbuf=576056 // 2,  # 576 kB
                            size_regf=1024 // 2,  # 1 kB
                           )

        cost = Cost(mac_op=2e-12,
                    mem_hier=(240e-12, 28e-12, 4e-12, 1e-12),  # pJ/16-b
                    noc_hop=0,
                    unit_static=320e-12)

        nnd = NNDataflow(network, batch_size, resource, cost,
                         self.map_strategy)
        tops, _ = nnd.schedule_search(self.options)
        self.assertTrue(tops)
        dfsch_l1 = tops[0]

        ## T-16 configuration.

        resource = Resource(proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                                   dim=PhyDim2(4, 4),
                                                   type=NodeRegion.PROC),
                            data_regions=(NodeRegion(origin=PhyDim2(0, 0),
                                                     dim=PhyDim2(4, 4),
                                                     type=NodeRegion.DATA),),
                            dim_array=PhyDim2(14, 14),
                            size_gbuf=133032 // 2,  # 133 kB
                            size_regf=512 // 2,  # 512 B
                           )

        cost = Cost(mac_op=2e-12,
                    mem_hier=(80e-12, 14e-12, 4e-12, 0.6e-12),  # pJ/16-b
                    noc_hop=40e-12,
                    unit_static=200e-12)

        options = Option(sw_gbuf_bypass=(True, True, True),
                         sw_solve_loopblocking=True,
                         partition_hybrid=True)

        nnd = NNDataflow(network, batch_size, resource, cost,
                         self.map_strategy)
        tops, _ = nnd.schedule_search(options)
        self.assertTrue(tops)
        dfsch_t16 = tops[0]

        ## Check results.

        # Same workload.
        self.assertAlmostEqual(dfsch_t16.total_ops, dfsch_l1.total_ops)

        # Performance of T-16 is proportional to PE resource (20% margin).
        self.assertLess(dfsch_t16.total_time,
                        1.2 * dfsch_l1.total_time * (16 * 16) / (14 * 14 * 16))
        # Energy reduced by > 30%.
        self.assertLess(dfsch_t16.total_cost, dfsch_l1.total_cost * 0.7)

