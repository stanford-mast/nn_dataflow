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

from nn_dataflow import InputLayer, FCLayer, PoolingLayer
from nn_dataflow import InterLayerPipeline
from nn_dataflow import Network
from nn_dataflow import NodeRegion
from nn_dataflow import PhyDim2
from nn_dataflow import Resource

from examples import import_network

class TestInterLayerPipeline(unittest.TestCase):
    ''' Tests for InterLayerPipeline. '''

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
        net.add('4p1', PoolingLayer(1, 1, 1), prevs='4')
        net.add('11', FCLayer(1, 1))
        net.add('12', FCLayer(1, 1))
        net.add('4p2', PoolingLayer(1, 1, 1), prevs='4')
        net.add('13', FCLayer(1, 1))
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
        for net_name in ['zfnet', 'vgg_net']:
            self.net[net_name] = import_network(net_name)

        self.resource = Resource(
            proc_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(8, 8),
                                   type=NodeRegion.PROC),
            data_regions=(NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(8, 8),
                                     type=NodeRegion.DATA),),
            dim_array=PhyDim2(16, 16), size_gbuf=65536, size_regf=64)

    def test_valid_args(self):
        ''' Valid arguments. '''
        ilp = InterLayerPipeline(self.net['net1'], self.resource)
        self.assertIs(ilp.network, self.net['net1'])
        self.assertIs(ilp.resource, self.resource)

    def test_invalid_network(self):
        ''' Invalid network. '''
        with self.assertRaisesRegexp(TypeError,
                                     'InterLayerPipeline: .*network.*'):
            _ = InterLayerPipeline(self.net['net1'].input_layer(),
                                   self.resource)

    def test_invalid_resource(self):
        ''' Invalid resource. '''
        with self.assertRaisesRegexp(TypeError,
                                     'InterLayerPipeline: .*resource.*'):
            _ = InterLayerPipeline(self.net['net1'], PhyDim2(1, 1))

    def test_topological_order(self):
        ''' Topological order. '''
        for net in self.net.values():

            if not net.net_name.startswith('net'):
                continue

            ilp = InterLayerPipeline(net, self.resource)

            for layer in net:
                vidx = ilp.dag_vertex_dict[layer]

                self.assertIn(layer, ilp.dag_vertex_list[vidx])

                # Layer is named by topological order.
                self.assertTrue(layer.startswith(str(vidx)))

            # Disjoint union.
            vs_list = [set(v) for v in ilp.dag_vertex_list]

            for idx, vs in enumerate(vs_list):
                for vs2 in vs_list[:idx]:
                    self.assertTrue(vs.isdisjoint(vs2))
            self.assertSetEqual(set.union(*vs_list), set(net))

    def test_prev(self):
        ''' Previous relationship. '''
        for net in self.net.values():

            ilp = InterLayerPipeline(net, self.resource)

            for vidx, prevs in ilp.dag_prev_dict.items():

                # Previous layers of the current vertex.
                prev_layers = set()
                v = ilp.dag_vertex_list[vidx]
                for l in v:
                    prev_layers.update(net.prev_layers(l)[0])
                prev_layers.difference_update(v)

                for pvidx in prevs:

                    # Previous vertices should be ordered before this vertex.
                    self.assertLess(pvidx, vidx)

                    # Previous vertex should have at least one previous layer.
                    if pvidx < 0:
                        self.assertIn(None, prev_layers)
                    else:
                        pv = ilp.dag_vertex_list[pvidx]
                        self.assertFalse(prev_layers.isdisjoint(pv))

    def test_next(self):
        ''' Next relationship. '''
        for net in self.net.values():

            ilp = InterLayerPipeline(net, self.resource)

            for vidx, nexts in ilp.dag_next_dict.items():

                # Next layers of the current vertex.
                next_layers = set()
                if vidx < 0:
                    next_layers = set(net.first_layers())
                else:
                    v = ilp.dag_vertex_list[vidx]
                    for l in v:
                        next_layers.update(net.next_layers(l))
                    next_layers.difference_update(v)

                for nvidx in nexts:

                    # Next vertices should be ordered after this vertex.
                    self.assertGreater(nvidx, vidx)

                    # Next vertex should have at least one next layer.
                    nv = ilp.dag_vertex_list[nvidx]
                    self.assertFalse(next_layers.isdisjoint(nv))

    def test_match_prev_next(self):
        ''' Previous and next relationships match. '''
        for net in self.net.values():

            ilp = InterLayerPipeline(net, self.resource)

            for vidx, prevs in ilp.dag_prev_dict.items():
                for pvidx in prevs:
                    self.assertIn(vidx, ilp.dag_next_dict[pvidx])

            for vidx, nexts in ilp.dag_next_dict.items():
                for nvidx in nexts:
                    self.assertIn(vidx, ilp.dag_prev_dict[nvidx])

    def test_ops(self):
        ''' Number of ops. '''
        for net in self.net.values():

            ilp = InterLayerPipeline(net, self.resource)

            self.assertEqual(sum(ilp.dag_vertex_ops),
                             sum(net[l].total_ops() for l in net))

    def test_gen_segment(self):
        ''' _gen_segment. '''

        # Simple case.
        ilp = InterLayerPipeline(self.net['net1'], self.resource)
        self.assertEqual(len(list(self._gen_all_segment(ilp))),
                         2 ** len(ilp.dag_vertex_list) - 1)

        # Linear case.
        # Number of different segments of n = 2 ** n - 1.
        # Number of compositions of n = 2 ** (n - 1).
        ilp = InterLayerPipeline(self.net['net2'], self.resource)
        cnt1 = 0
        cnt2 = 0
        for _, segment in self._gen_all_segment(ilp):
            cnt1 += 1
            if len(self.net['net2']) - 1 in segment:
                # The last segment, defines a composition.
                cnt2 += 1
        self.assertEqual(cnt1, 2 ** len(ilp.dag_vertex_list) - 1)
        self.assertEqual(cnt2, 2 ** (len(ilp.dag_vertex_list) - 1))

        # Fork case.
        ilp = InterLayerPipeline(self.net['net4'], self.resource)
        seg_list = list(seg for _, seg in self._gen_all_segment(ilp))
        seg_set = set(seg_list)
        self.assertEqual(len(seg_list), 2319)
        self.assertEqual(len(seg_set), 34)

        # Multiple first layers.
        self.assertGreater(len(self.net['net3'].first_layers()), 1)
        ilp = InterLayerPipeline(self.net['net3'], self.resource)
        seg_set = set(seg_list)
        self.assertIn((0,), seg_set)
        self.assertIn((1,), seg_set)

        # Segments are valid.
        for net in self.net.values():
            ilp = InterLayerPipeline(net, self.resource)
            seg_list = []

            for idx, segment in self._gen_all_segment(ilp):
                # Segment index, sequential or back-trace.
                self.assertLessEqual(idx, len(seg_list))

                # Segment, consecutive layers.
                self.assertTupleEqual(segment, tuple(range(min(segment),
                                                           max(segment) + 1)))

                seg_list = seg_list[0:idx]
                if seg_list:
                    # Compare against previous segment.
                    prev_segment = seg_list[-1]
                    self.assertEqual(max(prev_segment) + 1, min(segment))

                seg_list.append(segment)

        # Verify rules.
        ilp = InterLayerPipeline(self.net['net5'], self.resource)
        seg_sets = set(seg for _, seg in self._gen_all_segment(ilp))
        # Layers with no shared dependencies.
        self.assertNotIn((2, 3, 4), seg_sets)
        self.assertNotIn((8, 9), seg_sets)
        # Multiple previous layers.
        self.assertNotIn((5, 6, 7), seg_sets)
        self.assertNotIn((8, 9, 10), seg_sets)
        self.assertNotIn((10, 11, 12), seg_sets)
        # Multiple next layers.
        self.assertNotIn((0, 1, 2, 3), seg_sets)
        self.assertNotIn((3, 4), seg_sets)
        self.assertIn((3, 4, 5), seg_sets)
        self.assertNotIn((10, 11), seg_sets)

        # Real networks.
        ilp = InterLayerPipeline(self.net['zfnet'], self.resource)
        self.assertEqual(len(ilp.dag_vertex_list), 8)
        seg_list = list(seg for _, seg in self._gen_all_segment(ilp))
        seg_set = set(seg_list)
        self.assertEqual(len(seg_list), 2 ** len(ilp.dag_vertex_list) - 1)
        self.assertEqual(len(seg_set), 36)

        ilp = InterLayerPipeline(self.net['vgg_net'], self.resource)
        self.assertEqual(len(ilp.dag_vertex_list), 16)
        seg_list = list(seg for _, seg in self._gen_all_segment(ilp))
        seg_set = set(seg_list)
        self.assertEqual(len(seg_list), 2 ** len(ilp.dag_vertex_list) - 1)
        self.assertEqual(len(seg_set), 136)

    def test_gen_segment_no_repeating(self):
        ''' _gen_segment no_repeating. '''

        for net in self.net.values():

            ilp = InterLayerPipeline(net, self.resource)

            seg_list_rep = [seg for _, seg
                            in self._gen_all_segment(ilp, False)]

            ilp.seg_vertex_done.clear()
            seg_list_norep = [seg for _, seg
                              in self._gen_all_segment(ilp, True)]

            self.assertLess(len(seg_list_norep), len(seg_list_rep))
            self.assertSetEqual(set(seg_list_norep), set(seg_list_rep))
            self.assertEqual(len(seg_list_norep), len(set(seg_list_rep)))

        # Large networks with forks.
        for net_name in ['googlenet', 'resnet152']:
            net = import_network(net_name)

            ilp = InterLayerPipeline(net, self.resource)
            seg_list_norep = [seg for _, seg in self._gen_all_segment(ilp, True)]

            self.assertEqual(len(seg_list_norep), len(set(seg_list_norep)))

            # The number of different segments is between one and three times
            # of the number of layers.
            self.assertGreater(len(seg_list_norep), len(net))
            self.assertLessEqual(len(seg_list_norep), len(net) * 3)

    def test_allocate_segment(self):
        ''' _allocate_segment. '''
        # pylint: disable=protected-access

        net = self.net['net1']
        ilp = InterLayerPipeline(net, self.resource)
        self.assertListEqual(ilp.dag_vertex_ops, [200, 630, 1200, 2000])

        # Single vertex.
        for idx in range(len(ilp.dag_vertex_list)):
            segment = (idx,)
            seg, alloc = ilp._allocate_segment(segment)

            self.assertEqual(len(seg), 1)
            self.assertTupleEqual(seg[0], ilp.dag_vertex_list[idx])

            self.assertEqual(len(alloc), 1)
            self.assertEqual(len(alloc[0]), len(ilp.dag_vertex_list[idx]))
            for resource in alloc[0]:
                self.assertTupleEqual(resource.proc_region.origin, (0, 0))
                self.assertTupleEqual(resource.proc_region.dim,
                                      self.resource.proc_region.dim)

        # Multiple vertices.
        segment = (0, 1)
        _, alloc = ilp._allocate_segment(segment)
        nodes = self._subregion_num_nodes(alloc)
        self.assertTupleEqual(nodes, (16, 48))
        segment = (2, 3)
        _, alloc = ilp._allocate_segment(segment)
        nodes = self._subregion_num_nodes(alloc)
        self.assertTupleEqual(nodes, (24, 40))
        segment = (1, 2)
        _, alloc = ilp._allocate_segment(segment)
        nodes = self._subregion_num_nodes(alloc)
        self.assertTrue(nodes == (24, 40) or nodes == (22, 42))
        segment = (1, 2, 3)
        _, alloc = ilp._allocate_segment(segment)
        nodes = self._subregion_num_nodes(alloc)
        self.assertTrue(nodes == (12, 20, 32) or nodes == (10, 20, 34))

        # All segments.
        for _, segment in self._gen_all_segment(ilp, True):
            segalloc = ilp._allocate_segment(segment)
            if segalloc is None:
                continue
            seg, alloc = segalloc

            for vidx, ltpl in zip(segment, seg):
                self.assertTupleEqual(ltpl, ilp.dag_vertex_list[vidx])
            self._validate_allocation(ilp, seg, alloc)

            # This is a linear network structure.
            rlist = sum(alloc, ())

            # The data source of all layers except for the first in the
            # segment should be previous processing regions.
            for r in rlist[1:]:
                self.assertEqual(r.src_data_region().type, NodeRegion.PROC,
                                 'test_allocate_segment: data source '
                                 'should be PROC region.')

            # The data destination of all layers except for the last in the
            # segment should be local.
            for r in rlist[:-1]:
                self.assertEqual(r.dst_data_region().type, NodeRegion.PROC,
                                 'test_allocate_segment: data destination '
                                 'should be PROC region.')

        # Real network.
        net = self.net['zfnet']
        ilp = InterLayerPipeline(net, self.resource)

        for _, segment in self._gen_all_segment(ilp, True):
            segalloc = ilp._allocate_segment(segment, max_util_drop=0.1)
            if segalloc is None:
                continue
            seg, alloc = segalloc

            for vidx, ltpl in zip(segment, seg):
                self.assertTupleEqual(ltpl, ilp.dag_vertex_list[vidx])
            self._validate_allocation(ilp, seg, alloc)

            nodes = self._subregion_num_nodes(alloc)
            time = max(ilp.dag_vertex_ops[i] * 1. / n for i, n
                       in zip(segment, nodes))
            max_ops = time * sum(nodes)
            real_ops = sum(ilp.dag_vertex_ops[i] for i in segment)
            # Utilization.
            self.assertGreaterEqual(real_ops / max_ops, 0.9)

    @staticmethod
    def _gen_all_segment(ilp, no_repeating=False):
        # pylint: disable=protected-access
        return ilp._gen_segment(0, 0, set(), no_repeating=no_repeating)

    def _subregion_num_nodes(self, allocation):
        '''
        Get a tuple of numbers of nodes for the vertices in the segment
        allocation.
        '''
        nodes = []
        for alloc in allocation:
            self.assertTrue(all(isinstance(r, Resource) for r in alloc))
            n = alloc[0].proc_region.dim.size()
            self.assertTrue(all(r.proc_region.dim.size() == n
                                for r in alloc))
            nodes.append(n)
        return tuple(nodes)

    def _validate_allocation(self, ilp, layer_segment, allocation):
        ''' Validate segment resource allocation. '''

        # Number of nodes.
        nodes = self._subregion_num_nodes(allocation)
        self.assertEqual(sum(nodes), self.resource.proc_region.dim.size())

        # Used processing nodes.
        used_proc_nodes = set()

        # Layers that have data currently on-chip.
        data_regions = {}

        for ltpl, rtpl in zip(layer_segment, allocation):

            # Processing region.

            for n in rtpl[0].proc_region.node_iter():
                # FIXME: folded region.
                # self.assertTrue(self.resource.proc_region.contains_node(n),
                                # 'test_allocate_segment: node {} outside of '
                                # 'the processing region {}'
                                # .format(n, self.resource.proc_region))
                self.assertNotIn(n, used_proc_nodes,
                                 'test_allocate_segment: node {} has been '
                                 'used.'.format(n))
                used_proc_nodes.add(n)

            for jdx, (l, r) in enumerate(zip(ltpl, rtpl)):

                # Share processing region.
                self.assertEqual(rtpl[0].proc_region, r.proc_region)

                # Check data source.
                prev_layers, _ = ilp.network.prev_layers(l)

                for pl in prev_layers:
                    if pl not in data_regions:
                        # Previous layer is not on-chip, from memory.
                        self.assertEqual(
                            r.src_data_region(),
                            self.resource.src_data_region(),
                            'test_allocate_segment: layer {}\'s prev {} '
                            'is not on-chip, should be from {}, but {}.'
                            .format(l, pl, self.resource.src_data_region(),
                                    r.src_data_region()))
                    else:
                        # Previous layer is on-chip.
                        self.assertEqual(
                            r.src_data_region(), data_regions[pl],
                            'test_allocate_segment: layer {}\'s prev {} '
                            'is on-chip, should be from {}, but {}.'
                            .format(l, pl, data_regions[pl],
                                    r.src_data_region()))

                # Data destination.
                if r.dst_data_region() == self.resource.dst_data_region():
                    # Store back to memory.
                    pass
                else:
                    # Local.
                    self.assertEqual(r.dst_data_region(), r.proc_region,
                                     'test_allocate_segment: data can only '
                                     'be local if not storing back to mem.')
                    # Overwrite last local layer.
                    if jdx > 0:
                        self.assertEqual(data_regions.pop(ltpl[jdx - 1]),
                                         r.proc_region)
                    data_regions[l] = r.dst_data_region()

