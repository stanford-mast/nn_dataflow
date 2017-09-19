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

from nn_dataflow.core import InputLayer, FCLayer, PoolingLayer
from nn_dataflow.core import InterLayerPipeline
from nn_dataflow.core import Network
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import PipelineSegment
from nn_dataflow.core import Resource

from nn_dataflow.nns import import_network, all_networks

class TestPipelineFixture(unittest.TestCase):
    ''' Base fixture class for layer pipeline tests. '''

    def setUp(self):

        self.net = {}

        net = Network('net1')
        # Linear.
        net.set_input(InputLayer(10, 1))
        net.add('0', FCLayer(10, 20))
        net.add('1', FCLayer(20, 30))
        net.add('1p', PoolingLayer(30, 1, 1))
        net.add('2', FCLayer(30, 40))
        net.add('3', FCLayer(40, 50))
        self.net[net.net_name] = net

        net = Network('net2')
        # Long linear.
        net.set_input(InputLayer(1, 1))
        for idx in range(16):
            net.add(str(idx), FCLayer(1, 1))
        self.net[net.net_name] = net

        net = Network('net3')
        # Fork.
        # /0-2\   /6- 7- 8\
        #   x  4-5         12
        # \1-3/   \9-10-11/
        net.set_input(InputLayer(1, 1))
        net.add('0', FCLayer(1, 1), prevs=net.INPUT_LAYER_KEY)
        net.add('1', FCLayer(1, 1), prevs=net.INPUT_LAYER_KEY)
        net.add('2', FCLayer(2, 1), prevs=('0', '1'))
        net.add('2p', PoolingLayer(1, 1, 1))
        net.add('3', FCLayer(2, 1), prevs=('0', '1'))
        net.add('4', FCLayer(1, 1), prevs=('2p', '3'))
        net.add('5', FCLayer(1, 1))
        net.add('5p', PoolingLayer(1, 1, 1))
        net.add('6', FCLayer(1, 1), prevs='5p')
        net.add('7', FCLayer(1, 1))
        net.add('8', FCLayer(1, 1))
        net.add('9', FCLayer(1, 1), prevs='5p')
        net.add('10', FCLayer(1, 1))
        net.add('11', FCLayer(1, 1))
        net.add('12', FCLayer(1, 1), prevs=('8', '11'))
        self.net[net.net_name] = net

        net = Network('net4')
        # Complex fork.
        #          /5       \
        # 0-1-2-3-4-6-7-8-10-14
        #              \9/
        #          \11-12   /
        #          \13      /
        net.set_input(InputLayer(1, 1))
        net.add('0', FCLayer(1, 1))
        net.add('1', FCLayer(1, 1))
        net.add('2', FCLayer(1, 1))
        net.add('3', FCLayer(1, 1))
        net.add('4', FCLayer(1, 1))
        net.add('5', FCLayer(1, 1), prevs='4')
        net.add('6', FCLayer(1, 1), prevs='4')
        net.add('7', FCLayer(1, 1))
        net.add('8', FCLayer(1, 1), prevs='7')
        net.add('9', FCLayer(1, 1), prevs='7')
        net.add('10', FCLayer(1, 1))
        net.add('10p', PoolingLayer(1, 1, 1), prevs=('8', '10'))
        net.add('11', PoolingLayer(1, 1, 1), prevs='4')
        net.add('12', FCLayer(1, 1))
        net.add('13', PoolingLayer(1, 1, 1), prevs='4')
        net.add('14', FCLayer(1, 1), prevs=('5', '10p', '12', '13'))
        self.net[net.net_name] = net

        net = Network('net5')
        # Corner cases.
        #  ----\
        # //1-2\ 7-8\
        # 0-3-4-x   10-11-12
        #  \ \5/ 9 /  \__/
        #   6--/
        net.set_input(InputLayer(1, 1))
        net.add('0', FCLayer(1, 1))
        net.add('1', FCLayer(1, 1), prevs='0')
        net.add('2', FCLayer(1, 1))
        net.add('3', FCLayer(1, 1), prevs='0')
        net.add('4', FCLayer(1, 1), prevs='3')
        net.add('5', FCLayer(1, 1), prevs='3')
        net.add('6', FCLayer(1, 1), prevs='0')
        net.add('7', FCLayer(1, 1), prevs=('0', '2', '4', '5', '6'))
        net.add('8', FCLayer(1, 1))
        net.add('9', FCLayer(1, 1), prevs=('0', '2', '4', '5', '6'))
        net.add('10', FCLayer(1, 1), prevs=('8', '9'))
        net.add('11', FCLayer(1, 1))
        net.add('12', FCLayer(1, 1), prevs=('10', '11'))
        self.net[net.net_name] = net

        # Real networks.
        for net_name in all_networks():
            self.net[net_name] = import_network(net_name)

        self.batch_size = 16

        self.resource = Resource(
            proc_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(8, 8),
                                   type=NodeRegion.PROC),
            data_regions=(NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(8, 4),
                                     type=NodeRegion.DATA),
                          NodeRegion(origin=PhyDim2(0, 4), dim=PhyDim2(8, 4),
                                     type=NodeRegion.DATA)),
            dim_array=PhyDim2(16, 16), size_gbuf=65536, size_regf=64)


    def _make_ilp(self, network):
        ''' Make an InterLayerPipeline instance. '''
        return InterLayerPipeline(network, self.batch_size, self.resource)

    def _make_segment(self, vseg, network, temporal=False, max_util_drop=None):
        ''' Convert vertex segment to (layer) segment. '''
        kwargs = {}
        if max_util_drop is not None:
            kwargs['max_util_drop'] = max_util_drop
        ilp = self._make_ilp(network)
        seg = tuple(ilp.dag_vertex_list[vidx] for vidx in vseg)
        if temporal:
            seg = (sum(seg, tuple()),)
        return PipelineSegment(seg, ilp.network, ilp.batch_size, ilp.resource,
                               **kwargs)

    def _gen_all_segment(self, network, **kwargs):
        '''
        Generate all segments directly from all layers and all vertex segments.
        '''
        # pylint: disable=protected-access
        ilp = self._make_ilp(network)
        for layer in network:
            yield PipelineSegment(((layer,),), ilp.network, ilp.batch_size,
                                  ilp.resource)
        for vseg in ilp._gen_vseg():
            yield self._make_segment(vseg, network, **kwargs)

    def _validate_allocation(self, segment, allocation):
        ''' Validate segment resource allocation. '''

        # Match segment.
        self.assertEqual(len(allocation), len(segment))
        for ltpl, rtpl in zip(segment, allocation):
            self.assertEqual(len(rtpl), len(ltpl))
            self.assertTrue(all(isinstance(r, Resource) for r in rtpl))

        # Number of nodes.
        nodes = []  # number of nodes.
        for rtpl in allocation:
            nodes.append(rtpl[0].proc_region.dim.size())
        self.assertEqual(sum(nodes), self.resource.proc_region.dim.size())

        # Temporal schedules share processing region; spatial schedules use
        # non-overlapped processing regions.
        used_proc_nodes = set()  # used processing nodes
        for rtpl in allocation:
            proc_region = rtpl[0].proc_region
            self.assertTrue(all(r.proc_region == proc_region for r in rtpl))
            for n in proc_region.node_iter():
                self.assertTrue(self.resource.proc_region.contains_node(n),
                                '_validate_allocation: node {} outside of '
                                'the processing region {}'
                                .format(n, self.resource.proc_region))
                self.assertNotIn(n, used_proc_nodes,
                                 '_validate_allocation: node {} has been '
                                 'used.'.format(n))
                used_proc_nodes.add(n)

        # Data liveness.
        data_regions = {}  # layers that have data currently on-chip
        for ltpl, rtpl in zip(segment, allocation):

            for l, r in zip(ltpl, rtpl):

                # Check data source.
                prev_layers, _ = segment.network.prev_layers(l)

                for pl in prev_layers:
                    if pl not in data_regions:
                        # Previous layer is not on-chip, from memory.
                        self.assertEqual(
                            r.src_data_region(),
                            self.resource.src_data_region(),
                            '_validate_allocation: layer {}\'s prev {} '
                            'is not on-chip, should be from {}, but {}.'
                            .format(l, pl, self.resource.src_data_region(),
                                    r.src_data_region()))
                    elif data_regions[pl] != r.proc_region:
                        # Previous layer is on-chip and not local.
                        self.assertEqual(
                            r.src_data_region(), data_regions[pl],
                            '_validate_allocation: layer {}\'s prev {} '
                            'is on-chip, should be from {}, but {}.'
                            .format(l, pl, data_regions[pl],
                                    r.src_data_region()))

                # Update data based on destination.
                if r.dst_data_region() == self.resource.dst_data_region():
                    # Store back to memory.
                    pass
                else:
                    # Local.
                    self.assertEqual(r.dst_data_region(), r.proc_region,
                                     '_validate_allocation: data can only '
                                     'be local if not storing back to mem.')
                    # Overwrite.
                    local_node_set = set(r.dst_data_region().node_iter())
                    data_regions = {pl: data_regions[pl] for pl in data_regions
                                    if local_node_set.isdisjoint(
                                        data_regions[pl].node_iter())}
                    data_regions[l] = r.dst_data_region()

