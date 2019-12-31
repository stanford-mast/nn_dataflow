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

import re

from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer
from nn_dataflow.core import InterLayerPipeline
from nn_dataflow.core import Network
from nn_dataflow.core import Option
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import PipelineSegment

from . import TestPipelineFixture

class TestInterLayerPipeline(TestPipelineFixture):
    ''' Tests for InterLayerPipeline. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        ilp = InterLayerPipeline(self.net['net1'], self.batch_size,
                                 self.resource, max_util_drop=0.1)
        self.assertIs(ilp.network, self.net['net1'])
        self.assertEqual(ilp.batch_size, self.batch_size)
        self.assertIs(ilp.resource, self.resource)
        self.assertEqual(ilp.max_util_drop, 0.1)

    def test_invalid_network(self):
        ''' Invalid network. '''
        with self.assertRaisesRegex(TypeError,
                                    'InterLayerPipeline: .*network.*'):
            _ = InterLayerPipeline(self.net['net1'].input_layer(),
                                   self.batch_size, self.resource)

    def test_invalid_resource(self):
        ''' Invalid resource. '''
        with self.assertRaisesRegex(TypeError,
                                    'InterLayerPipeline: .*resource.*'):
            _ = InterLayerPipeline(self.net['net1'], self.batch_size,
                                   PhyDim2(1, 1))

    def test_invalid_max_util_drop(self):
        ''' Invalid max_util_drop. '''
        with self.assertRaisesRegex(ValueError,
                                    'InterLayerPipeline: .*max_util_drop.*'):
            _ = InterLayerPipeline(self.net['net1'], self.batch_size,
                                   self.resource, max_util_drop=1.1)

        with self.assertRaisesRegex(ValueError,
                                    'InterLayerPipeline: .*max_util_drop.*'):
            _ = InterLayerPipeline(self.net['net1'], self.batch_size,
                                   self.resource, max_util_drop=-0.1)

    def test_topological_order(self):
        ''' Topological order. '''
        for net in self.net.values():

            if not net.net_name.startswith('net'):
                continue

            ilp = self._make_ilp(net)

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

    def test_vertex_no_merge_lr(self):
        ''' LocalRegionLayer has no previous layer to merge with. '''
        net = Network('tmp_net')
        net.set_input_layer(InputLayer(30, 1))
        net.add('0', PoolingLayer(30, 1, 1))
        net.add('1', FCLayer(30, 40))
        net.add('1p', PoolingLayer(40, 1, 1))

        ilp = self._make_ilp(net)

        for layer in net:
            vidx = ilp.dag_vertex_dict[layer]

            self.assertIn(layer, ilp.dag_vertex_list[vidx])

            # Layer is named by topological order.
            self.assertTrue(layer.startswith(str(vidx)))

    def test_prev(self):
        ''' Previous relationship. '''
        for net in self.net.values():

            ilp = self._make_ilp(net)

            for vidx, prevs in ilp.dag_prev_dict.items():

                # Previous layers of the current vertex.
                prev_layers = set()
                v = ilp.dag_vertex_list[vidx]
                for l in v:
                    prev_layers.update(net.prevs(l))
                prev_layers.difference_update(v)

                for pvidx in prevs:

                    # Previous vertices should be ordered before this vertex.
                    self.assertLess(pvidx, vidx)

                    # Previous vertex should have at least one previous layer.
                    if pvidx < 0:
                        self.assertTrue(
                            None in prev_layers
                            or not prev_layers.isdisjoint(net.ext_layers()))
                    else:
                        pv = ilp.dag_vertex_list[pvidx]
                        self.assertFalse(prev_layers.isdisjoint(pv))

    def test_next(self):
        ''' Next relationship. '''
        for net in self.net.values():

            ilp = self._make_ilp(net)

            for vidx, nexts in ilp.dag_next_dict.items():

                # Next layers of the current vertex.
                next_layers = set()
                if vidx < 0:
                    # Go through all layers and add those with input layer as
                    # previous.
                    for l in net:
                        prevs = set(net.prevs(l))
                        if None in prevs \
                                or not prevs.isdisjoint(net.ext_layers()):
                            next_layers.add(l)
                else:
                    v = ilp.dag_vertex_list[vidx]
                    for l in v:
                        next_layers.update(net.nexts(l))
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

            ilp = self._make_ilp(net)

            for vidx, prevs in ilp.dag_prev_dict.items():
                for pvidx in prevs:
                    self.assertIn(vidx, ilp.dag_next_dict[pvidx])

            for vidx, nexts in ilp.dag_next_dict.items():
                for nvidx in nexts:
                    self.assertIn(vidx, ilp.dag_prev_dict[nvidx])

    def test_gen_vseg(self):
        ''' _gen_vseg. '''
        # pylint: disable=protected-access

        # Simple case.
        ilp = self._make_ilp(self.net['net1'])
        num = len(ilp.dag_vertex_list)
        self.assertEqual(len(list(ilp._gen_vseg())),
                         (num + 1) * num // 2)

        # Linear case.
        # Number of different vsegs of n = 1 + ... + n
        ilp = self._make_ilp(self.net['net2'])
        num = len(ilp.dag_vertex_list)
        self.assertEqual(len(list(ilp._gen_vseg())),
                         (num + 1) * num // 2)

        # Fork case.
        ilp = self._make_ilp(self.net['net4'])
        vseg_list = list(ilp._gen_vseg())
        self.assertEqual(len(vseg_list), 39)
        # Case with one of multiple previous vertices on-chip.
        self.assertIn((9, 10), vseg_list)
        self.assertIn((13, 14), vseg_list)
        # Case with only one next vertex off-chip.
        self.assertIn((7, 8), vseg_list)
        self.assertNotIn((4, 5, 6), vseg_list)

        # Multiple first layers.
        self.assertGreater(len(self.net['net3'].firsts()), 1)
        ilp = self._make_ilp(self.net['net3'])
        vseg_list = list(ilp._gen_vseg())
        self.assertIn((0,), vseg_list)
        self.assertIn((1,), vseg_list)

        # Verify rules.
        ilp = self._make_ilp(self.net['net5'])
        vseg_list = list(ilp._gen_vseg())
        # Layers with no shared dependencies.
        self.assertNotIn((2, 3, 4), vseg_list)
        self.assertNotIn((8, 9), vseg_list)
        # Multiple previous layers.
        self.assertNotIn((5, 6, 7), vseg_list)
        self.assertNotIn((8, 9, 10), vseg_list)
        self.assertNotIn((10, 11, 12), vseg_list)
        # Multiple next layers.
        self.assertNotIn((0, 1, 2, 3), vseg_list)
        self.assertIn((3, 4), vseg_list)
        self.assertIn((3, 4, 5), vseg_list)
        self.assertIn((10, 11), vseg_list)

        # No duplicate.
        for net in self.net.values():
            ilp = self._make_ilp(net)
            vseg_list = list(ilp._gen_vseg())
            self.assertEqual(len(vseg_list), len(set(vseg_list)))

        # Real networks.
        ilp = self._make_ilp(self.net['zfnet'])
        self.assertEqual(len(ilp.dag_vertex_list), 8)
        vseg_list = list(ilp._gen_vseg())
        self.assertEqual(len(vseg_list), 36)

        ilp = self._make_ilp(self.net['vgg_net'])
        self.assertEqual(len(ilp.dag_vertex_list), 16)
        vseg_list = list(ilp._gen_vseg())
        self.assertEqual(len(vseg_list), 136)

        # Large networks with forks.
        for net_name in ['googlenet', 'resnet152']:
            net = self.net[net_name]

            ilp = self._make_ilp(net)
            vseg_list = list(ilp._gen_vseg())
            self.assertEqual(len(vseg_list), len(set(vseg_list)))

            # The number of different vsegs is between one and eight times of
            # the number of layers.
            self.assertGreater(len(vseg_list), len(net))
            self.assertLessEqual(len(vseg_list), len(net) * 8)

    def test_gen_vseg_twice(self):
        ''' _gen_vseg twice. '''
        # pylint: disable=protected-access
        for net_name in self.net:
            if not net_name.startswith('net'):
                continue

            net = self.net[net_name]
            ilp = self._make_ilp(net)

            vseg_list_1 = list(ilp._gen_vseg())
            vseg_list_2 = list(ilp._gen_vseg())

            self.assertListEqual(vseg_list_1, vseg_list_2)

    def test_ordered_layer_list(self):
        ''' Get ordered_layer_list. '''

        # https://stackoverflow.com/a/4836734/5277823
        nat_key = lambda key: tuple(int(c) if c.isdigit() else c.lower()
                                    for c in re.split('([0-9]+)', key))

        for net_name in ['net1', 'net2', 'net3', 'net4', 'net5']:
            net = self.net[net_name]
            ilp = self._make_ilp(net)
            ord_list = ilp.ordered_layer_list()

            # In natural order.
            self.assertTrue(all(nat_key(l1) < nat_key(l2) for l1, l2
                                in zip(ord_list, ord_list[1:])))

    def test_gen_segment(self):
        ''' gen_segment(). '''
        for net_name in self.net:
            net = self.net[net_name]
            ilp = self._make_ilp(net)

            # No pipelining.
            options = Option()
            segs_n_lst = list(ilp.gen_segment(options))
            segs_n = set(segs_n_lst)
            self.assertEqual(len(segs_n_lst), len(segs_n))
            for seg in segs_n:
                self.assertEqual(len(seg), 1)
                self.assertEqual(len(seg[0]), 1)
                self.assertIn(seg[0][0], net)

            # Spatial pipelining.
            options = Option(partition_interlayer=True)
            segs_sp_lst = list(ilp.gen_segment(options))
            segs_sp = set(segs_sp_lst)
            self.assertEqual(len(segs_sp_lst), len(segs_sp))
            for seg in segs_sp:
                for ltpl in seg:
                    self.assertLessEqual(sum(1 for l in ltpl
                                             if isinstance(l, ConvLayer)),
                                         1)
            self.assertTrue(segs_sp.issuperset(segs_n))

            # Temporal pipelining.
            options = Option(hw_gbuf_save_writeback=True)
            segs_tp_lst = list(ilp.gen_segment(options))
            segs_tp = set(segs_tp_lst)
            self.assertEqual(len(segs_tp_lst), len(segs_tp))
            for seg in segs_tp:
                self.assertEqual(len(seg), 1)
            self.assertTrue(segs_tp.issuperset(segs_n))

            # Spatial and temporal pipelining.
            options = Option(partition_interlayer=True,
                             hw_gbuf_save_writeback=True)
            segs_stp_lst = list(ilp.gen_segment(options))
            segs_stp = set(segs_stp_lst)
            self.assertEqual(len(segs_stp_lst), len(segs_stp))
            self.assertSetEqual(segs_stp, segs_tp | segs_sp)
            # Only single-layer and single-vertex segments have the same
            # spatial and temporal pipelining.
            segs_intersect = segs_tp & segs_sp
            segs_single = segs_n
            segs_single |= set(PipelineSegment((v,), ilp.network,
                                               ilp.batch_size, ilp.resource)
                               for v in ilp.dag_vertex_list)
            self.assertTrue(segs_intersect.issubset(segs_single))

    def test_gen_segment_max_degree(self):
        ''' gen_segment() maximum degree. '''
        net = self.net['vgg_net']
        ilp = self._make_ilp(net)

        options = Option(partition_interlayer=True,
                         hw_gbuf_save_writeback=True,
                         layer_pipeline_max_degree=4)
        for segment in ilp.gen_segment(options):
            self.assertLessEqual(sum(1 if isinstance(net[l], ConvLayer) else 0
                                     for ltpl in segment for l in ltpl),
                                 4)

    def test_gen_segment_vseg(self):
        ''' gen_segment() vertex segment. '''

        for net_name in self.net:
            if not net_name.startswith('net'):
                continue
            net = self.net[net_name]

            ilp = self._make_ilp(net)
            options = Option(partition_interlayer=True)

            seg_set = set(ilp.gen_segment(options))
            self.assertTrue(seg_set)

            seg_v_set = set(self._gen_all_segment(net))
            self.assertTrue(seg_set.issubset(seg_v_set))

    def test_gen_segment_multi_prevs(self):
        ''' gen_segment() with multiple previous vertices. '''
        # pylint: disable=protected-access

        net = self.net['net4']
        ilp = self._make_ilp(net)

        vseg_set = set(ilp._gen_vseg())
        self.assertIn((9, 10), vseg_set)
        self.assertIn((13, 14), vseg_set)

        options = Option(partition_interlayer=True)
        seg_set = set(ilp.gen_segment(options))

        # 10 only has neighbor source 9; 10p only has local source 10 and
        # memory source 8. Valid.
        self.assertIn(self._make_segment((9, 10), ilp.network), seg_set)
        # 14 has both neighbor source 13, and memory source 12, etc.. Invalid.
        self.assertNotIn(self._make_segment((13, 14), ilp.network), seg_set)

    def test_gen_segment_one_nexts(self):
        ''' gen_segment() with missing one next vertex. '''
        # pylint: disable=protected-access

        net = self.net['net4']
        ilp = self._make_ilp(net)

        vseg_set = set(ilp._gen_vseg())
        self.assertIn((7, 8), vseg_set)
        self.assertNotIn((4, 5, 6), vseg_set)

        options = Option(partition_interlayer=True)
        seg_set = set(ilp.gen_segment(options))

        self.assertIn(self._make_segment((7, 8), ilp.network), seg_set)
        self.assertNotIn(self._make_segment((4, 5, 6), ilp.network), seg_set)

    def test_gen_segment_not_opt(self):
        ''' gen_segment() not with_opt. '''

        options_with_opt = Option(partition_interlayer=True,
                                  hw_gbuf_save_writeback=True,
                                  layer_pipeline_opt=True)
        options_not_opt = Option(partition_interlayer=True,
                                 hw_gbuf_save_writeback=True,
                                 layer_pipeline_opt=False)

        # Linear ones
        for net_name in ['net1', 'net2', 'zfnet']:
            net = self.net[net_name]
            ilp = self._make_ilp(net)

            segs_with_opt = set(seg.seg
                                for seg in ilp.gen_segment(options_with_opt))
            segs_not_opt = set(seg.seg
                               for seg in ilp.gen_segment(options_not_opt))

            self.assertSetEqual(segs_with_opt, segs_not_opt)

        # Non-linear ones
        for net_name in ['net3', 'net4', 'net5', 'net6', 'net7', 'googlenet']:
            net = self.net[net_name]
            ilp = self._make_ilp(net)

            segs_with_opt = set(seg.seg
                                for seg in ilp.gen_segment(options_with_opt))
            segs_not_opt = set(seg.seg
                               for seg in ilp.gen_segment(options_not_opt))

            self.assertTrue(segs_with_opt.issuperset(segs_not_opt))

    def test_gen_segment_resnet(self):
        ''' gen_segment() with ResNet. '''

        net = self.net['resnet152']
        ilp = self._make_ilp(net)

        options = Option(partition_interlayer=True)

        # One residual module fits.
        segment = PipelineSegment(
            (('conv3_2_a',), ('conv3_2_b',), ('conv3_2_c', 'conv3_2_res')),
            ilp.network, ilp.batch_size, ilp.resource)

        self.assertTupleEqual(net.prevs('conv3_2_res'),
                              ('conv3_1_res', 'conv3_2_c'))
        self.assertTrue(segment.valid)

        segs = set(seg.seg for seg in ilp.gen_segment(options))
        self.assertIn(segment.seg, segs)

    def test_gen_segment_lstm(self):
        ''' gen_segment() with LSTM cell. '''

        net = self.net['lstm_phoneme']
        ilp = self._make_ilp(net)

        options = Option(partition_interlayer=True)

        # Find a cell.
        cname = None
        for l in net:
            if l[-6:] == '_igate':
                cname = l[:-6]
        self.assertIsNotNone(cname)

        # One LSTM cell fits.
        segment = PipelineSegment(
            ((cname + '_cand',),
             (cname + '_igate', cname + '_cout_i'),
             (cname + '_fgate', cname + '_cout_f', cname + '_cout'),
             (cname + '_ogate', cname + '_hout')),
            ilp.network, ilp.batch_size, ilp.resource)

        self.assertTrue(segment.valid)

        segs = set(seg.seg for seg in ilp.gen_segment(options))
        self.assertIn(segment.seg, segs)

