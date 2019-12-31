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

import itertools

from nn_dataflow.core import ConvLayer
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import PipelineSegment
from nn_dataflow.core import PipelineSegmentTiming

from . import TestPipelineFixture

class TestPipelineSegment(TestPipelineFixture):
    ''' Tests for PipelineSegment. '''

    # pylint: disable=too-many-public-methods

    def test_valid_args(self):
        ''' Valid arguments. '''
        segment = PipelineSegment((('0',), ('1', '1p')),
                                  self.net['net1'], self.batch_size,
                                  self.resource)
        self.assertTrue(segment.valid)
        self.assertTupleEqual(segment.seg, (('0',), ('1', '1p')))
        self.assertIs(segment.network, self.net['net1'])
        self.assertEqual(segment.batch_size, self.batch_size)
        self.assertIs(segment.resource, self.resource)

    def test_invalid_seg(self):
        ''' Invalid seg. '''
        with self.assertRaisesRegex(TypeError,
                                    'PipelineSegment: .*seg.*tuple.*'):
            _ = PipelineSegment([('0',), ('1', '1p')],
                                self.net['net1'], self.batch_size,
                                self.resource)

        with self.assertRaisesRegex(TypeError,
                                    'PipelineSegment: .*seg.*sub-tuple.*'):
            _ = PipelineSegment(('0', '1', '1p'),
                                self.net['net1'], self.batch_size,
                                self.resource)

    def test_invalid_network(self):
        ''' Invalid network. '''
        with self.assertRaisesRegex(TypeError,
                                    'PipelineSegment: .*network.*'):
            _ = PipelineSegment((('0',), ('1', '1p')),
                                self.net['net1'].input_layer(), self.batch_size,
                                self.resource)

    def test_invalid_resource(self):
        ''' Invalid resource. '''
        with self.assertRaisesRegex(TypeError,
                                    'PipelineSegment: .*resource.*'):
            _ = PipelineSegment((('0',), ('1', '1p')),
                                self.net['net1'], self.batch_size,
                                PhyDim2(1, 1))

    def test_init_deps_not_valid(self):
        ''' Not valid segment due to init deps. '''

        # Not utilize local data.
        segment = self._make_segment((0, 1), self.net['net3'], temporal=True)
        self.assertFalse(segment.valid)
        self.assertFalse(hasattr(segment, 'alloc'))

        # Local data not available.
        segment = self._make_segment((10, 11, 12), self.net['net5'],
                                     temporal=True)
        self.assertFalse(segment.valid)
        self.assertFalse(hasattr(segment, 'alloc'))

        # Multiple neighbor source in one spatial scheduling.
        segment = self._make_segment((1, 2), self.net['net8'])
        self.assertFalse(segment.valid)
        self.assertFalse(hasattr(segment, 'alloc'))

        # Both memory source and neighbor source.
        segment = self._make_segment((13, 14), self.net['net4'])
        self.assertFalse(segment.valid)
        self.assertFalse(hasattr(segment, 'alloc'))

        # Valid cases.

        # Both memory destination and neighbor destination.
        segment = self._make_segment((7, 8), self.net['net4'])
        self.assertTrue(segment.valid)

    def test_init_deps_not_opt(self):
        ''' Init deps for segment not with opt. '''

        # Multiple on-chip sources.
        segment = self._make_segment((3, 4), self.net['net8'])
        self.assertTrue(segment.valid)
        segment = self._make_segment((3, 4), self.net['net8'], with_opt=False)
        self.assertFalse(segment.valid)

        # Multiple on-chip destinations.
        segment = self._make_segment((4, 5, 6), self.net['net4'])
        self.assertTrue(segment.valid)
        segment = self._make_segment((4, 5, 6), self.net['net4'],
                                     with_opt=False)
        self.assertFalse(segment.valid)

        # Multiple linear chains.
        segment = self._make_segment((5, 6), self.net['net4'])
        self.assertTrue(segment.valid)
        segment = self._make_segment((5, 6), self.net['net4'], with_opt=False)
        self.assertFalse(segment.valid)

    def test_alloc_not_valid(self):
        ''' Not valid segment due to alloc. '''

        segment = self._make_segment((0, 1), self.net['net1'],
                                     max_util_drop=0.01)
        self.assertFalse(segment.valid)

    def test_as_sequence(self):
        ''' As a sequence. '''
        segment = self._make_segment((0, 1), self.net['net1'])
        self.assertTrue(segment.valid)

        self.assertSequenceEqual(segment, segment.seg)
        self.assertTupleEqual(tuple(segment), segment.seg)

        for ltpl in segment:
            for layer in ltpl:
                self.assertIn(layer, self.net['net1'])

    def test_equal(self):
        ''' Equality. '''
        seg1 = self._make_segment((0, 1), self.net['net1'], max_util_drop=0.1)
        seg2 = self._make_segment((0, 1), self.net['net1'], max_util_drop=0.01)
        seg3 = self._make_segment((0, 1), self.net['net1'], temporal=True)
        self.assertNotEqual(seg1, seg2)
        self.assertNotEqual(seg1, seg3)

        seg4 = self._make_segment((0, 1), self.net['net1'], max_util_drop=0.1)
        self.assertEqual(seg1, seg4)

        net = self.net['net1']
        self.assertSetEqual(set(self._gen_all_segment(net)),
                            set(itertools.chain(self._gen_all_segment(net),
                                                self._gen_all_segment(net))))

    def test_repr(self):
        ''' __repr__. '''
        seg = self._make_segment((0, 1), self.net['net1'], max_util_drop=0.1)
        str_ = repr(seg)
        self.assertIn(repr(seg.seg), str_)
        self.assertIn(repr(seg.resource), str_)
        self.assertIn(repr(seg.max_util_drop), str_)

    def test_alloc_proc(self):
        ''' _alloc_proc. '''
        # pylint: disable=protected-access

        net = self.net['net1']
        self.assertListEqual([net[l].total_ops() for l in net],
                             [200, 600, 30, 1200, 2000])

        ilp = self._make_ilp(net)

        # Single vertex.

        for idx in range(len(ilp.dag_vertex_list)):
            segment = self._make_segment((idx,), ilp.network)
            psr = segment._alloc_proc()

            self.assertEqual(len(psr), 1)
            self.assertTupleEqual(psr[0].origin, (0, 0))
            self.assertTupleEqual(psr[0].dim, self.resource.proc_region.dim)
            self.assertEqual(psr[0].type, NodeRegion.PROC)

        # Multiple vertices.

        psr = self._make_segment((0, 1), net)._alloc_proc()
        nodes = [nr.dim.size() for nr in psr]
        self.assertListEqual(nodes, [16, 48])

        psr = self._make_segment((2, 3), net)._alloc_proc()
        nodes = [nr.dim.size() for nr in psr]
        self.assertListEqual(nodes, [24, 40])

        psr = self._make_segment((1, 2), net)._alloc_proc()
        nodes = [nr.dim.size() for nr in psr]
        self.assertIn(nodes, ([24, 40], [22, 42]))

        psr = self._make_segment((1, 2, 3), net)._alloc_proc()
        nodes = [nr.dim.size() for nr in psr]
        self.assertIn(nodes, ([12, 20, 32], [10, 20, 34]))

        # All segments.

        def _check_all_segment(ilp):
            for vseg in ilp._gen_vseg():
                segment = self._make_segment(vseg, ilp.network)
                psr = segment._alloc_proc()
                if psr is None:
                    continue

                # Utilization.
                nodes = [nr.dim.size() for nr in psr]
                ops = [sum(ilp.network[l].total_ops() for l in ltpl)
                       for ltpl in segment]
                self.assertEqual(len(nodes), len(ops))
                time = max(o * 1. / n for o, n in zip(ops, nodes))
                max_ops = time * sum(nodes)
                real_ops = sum(ops)
                self.assertGreaterEqual(real_ops / max_ops, 0.9)

        _check_all_segment(ilp)

        for net_name in ['zfnet', 'net3']:
            net = self.net[net_name]
            ilp = self._make_ilp(net)
            _check_all_segment(ilp)

    def test_allocation(self):
        ''' allocation(). '''

        # Single vertex.

        net = self.net['net1']
        ilp = self._make_ilp(net)
        for idx in range(len(ilp.dag_vertex_list)):
            segment = self._make_segment((idx,), ilp.network)
            alloc = segment.allocation()
            self.assertIsNotNone(alloc)
            self._validate_allocation(segment, alloc)

        # Linear networks.

        for net_name in ['net1', 'net2']:

            net = self.net[net_name]

            for segment in self._gen_all_segment(net):

                alloc = segment.allocation()
                if alloc is None:
                    continue

                self._validate_allocation(segment, alloc)

                # This is a linear network structure.
                rlist = sum(alloc, tuple())

                # The data source of all layers except for the first in the
                # segment should be previous processing regions.
                for r in rlist[1:]:
                    self.assertEqual(r.src_data_region.type, NodeRegion.PROC,
                                     'test_segment_allocation: '
                                     'data source should be PROC region.')

                # The data destination of all layers except for the last in the
                # segment should be local.
                for r in rlist[:-1]:
                    self.assertEqual(r.dst_data_region.type, NodeRegion.PROC,
                                     'test_segment_allocation: '
                                     'data destination should be PROC region.')

        # Complex networks.

        for net_name in ['net3', 'net4', 'net5']:

            net = self.net[net_name]

            for segment in self._gen_all_segment(net):

                alloc = segment.allocation()
                if alloc is None:
                    continue

                self._validate_allocation(segment, alloc)

        # Real networks.

        for net_name in self.net:

            if net_name.startswith('net'):
                continue
            net = self.net[net_name]

            for segment in self._gen_all_segment(net):

                alloc = segment.allocation()
                if alloc is None:
                    continue

                self._validate_allocation(segment, alloc)

    def test_allocation_sh_mem_src(self):
        ''' allocation() shared mem src. '''

        net = self.net['net3']

        segment = self._make_segment((6, 7, 8, 9), net)
        self.assertTrue(segment.valid)

        alloc = segment.allocation()
        self.assertEqual(alloc[3][0].src_data_region, alloc[0][0].proc_region)

        segment = self._make_segment((6, 7, 8, 9), net, with_opt=False)
        self.assertFalse(segment.valid)

        net = self.net['net5']

        segment = self._make_segment((1, 2, 3), net)
        self.assertTrue(segment.valid)

        alloc = segment.allocation()
        self.assertEqual(alloc[2][0].src_data_region, alloc[0][0].proc_region)

        segment = self._make_segment((1, 2, 3), net, with_opt=False)
        self.assertFalse(segment.valid)

        net = self.net['net4']

        segment = self._make_segment((8, 9), net)
        self.assertTrue(segment.valid)

        alloc = segment.allocation()
        self.assertEqual(alloc[1][0].src_data_region, alloc[0][0].proc_region)

        segment = self._make_segment((8, 9), net, with_opt=False)
        self.assertFalse(segment.valid)

    def test_allocation_temp(self):
        ''' allocation() temporal. '''

        for net in self.net.values():

            for segment in self._gen_all_segment(net, temporal=True):

                alloc = segment.allocation()
                if alloc is None:
                    continue

                self._validate_allocation(segment, alloc)

    def test_allocation_no_time_mux(self):
        ''' allocation() no_time_mux. '''
        net = self.net['net2']

        segment = self._make_segment(tuple(range(16)), net)
        self.assertTrue(segment.valid)

        alloc = segment.allocation()
        self.assertTrue(all(r.no_time_mux for rtpl in alloc for r in rtpl))

        segment = self._make_segment(tuple(range(8)), net)
        self.assertTrue(segment.valid)

        alloc = segment.allocation()
        self.assertFalse(any(r.no_time_mux for rtpl in alloc for r in rtpl))

        segment = self._make_segment(tuple(range(16)), net, temporal=True)
        self.assertTrue(segment.valid)

        alloc = segment.allocation()
        self.assertFalse(any(r.no_time_mux for rtpl in alloc for r in rtpl))

    def test_allocation_invalid(self):
        ''' allocation() for invalid segment. '''
        segment = self._make_segment((0, 1), self.net['net3'], temporal=True)
        self.assertFalse(segment.valid)
        self.assertIsNone(segment.allocation())

    def test_gen_constraint(self):
        ''' gen_constraint(). '''

        # Single vertex.

        for net_name in self.net:

            net = self.net[net_name]
            ilp = self._make_ilp(net)

            for idx in range(len(ilp.dag_vertex_list)):
                segment = self._make_segment((idx,), ilp.network)
                self.assertTrue(segment.valid)

                for constraint, _ in segment.gen_constraint():
                    self._validate_constraint(segment, constraint)

                    # No top loop constraint for single-layer segment.
                    if len(constraint) == 1 and len(constraint[0]) == 1:
                        for c in itertools.chain.from_iterable(constraint):
                            self.assertTrue(c.topifm == 0 and c.topofm == 0
                                            and c.topbat == 0)

        # Spatial pipelining.

        for net_name in self.net:

            if not net_name.startswith('net') and net_name != 'zfnet':
                continue

            net = self.net[net_name]

            for segment in self._gen_all_segment(net):
                if not segment.valid:
                    continue

                for constraint, _ in segment.gen_constraint():
                    self._validate_constraint(segment, constraint)

        # Special cases.

        net = self.net['net2']

        segment = PipelineSegment((('0', '1'), ('2', '3')), net,
                                  self.batch_size, self.resource)

        for constraint, _ in segment.gen_constraint():
            self._validate_constraint(segment, constraint)

    def test_gen_constraint_fbofm_init(self):
        ''' gen_constraint() deciding fbofm_init. '''

        net = self.net['zfnet']

        # Two spatial, fbofm_init == False.
        segment = PipelineSegment((('fc2',), ('fc3',)),
                                  net, self.batch_size, self.resource)
        self.assertTrue(segment.valid)
        self.assertFalse(segment.cstr_symargs[0][0].get('fbofm', False))
        self.assertFalse(segment.cstr_symargs[1][0].get('fbifm', False))

        # Two spatial, fbofm_init == False.
        segment = PipelineSegment((('conv5', 'pool3'), ('fc1',)),
                                  net, self.batch_size, self.resource)
        self.assertTrue(segment.valid)
        self.assertFalse(segment.cstr_symargs[0][0].get('fbofm', False))
        self.assertFalse(segment.cstr_symargs[0][1].get('fbofm', False))
        self.assertFalse(segment.cstr_symargs[1][0].get('fbifm', False))

        # Four spatial, fbofm_init == False.
        segment = PipelineSegment((('conv1', 'pool1'), ('conv2', 'pool2'),
                                   ('conv3',), ('conv4',)),
                                  net, self.batch_size, self.resource)
        self.assertTrue(segment.valid)
        self.assertFalse(segment.cstr_symargs[0][0].get('fbofm', False))
        self.assertFalse(segment.cstr_symargs[0][1].get('fbofm', False))
        self.assertFalse(segment.cstr_symargs[1][0].get('fbifm', False))
        self.assertTrue(segment.cstr_symargs[1][0]['fbofm'])
        self.assertTrue(segment.cstr_symargs[1][1]['fbofm'])
        self.assertTrue(segment.cstr_symargs[2][0]['fbifm'])
        self.assertFalse(segment.cstr_symargs[2][0].get('fbofm', False))
        self.assertFalse(segment.cstr_symargs[3][0].get('fbifm', False))

        # Three spatial, fbofm_init == False.
        segment = PipelineSegment((('conv4',), ('conv5', 'pool3'), ('fc1',)),
                                  net, self.batch_size, self.resource)
        self.assertTrue(segment.valid)
        self.assertFalse(segment.cstr_symargs[0][0].get('fbofm', False))
        self.assertFalse(segment.cstr_symargs[1][0].get('fbifm', False))
        self.assertTrue(segment.cstr_symargs[1][0]['fbofm'])
        self.assertTrue(segment.cstr_symargs[1][1]['fbofm'])
        self.assertTrue(segment.cstr_symargs[2][0]['fbifm'])

        # Three spatial, fbofm_init == False.
        segment = PipelineSegment((('conv2', 'pool2'), ('conv3',), ('conv4',)),
                                  net, self.batch_size, self.resource)
        self.assertTrue(segment.valid)
        self.assertFalse(segment.cstr_symargs[0][0].get('fbofm', False))
        self.assertFalse(segment.cstr_symargs[0][1].get('fbofm', False))
        self.assertFalse(segment.cstr_symargs[1][0].get('fbifm', False))
        self.assertTrue(segment.cstr_symargs[1][0]['fbofm'])
        self.assertTrue(segment.cstr_symargs[2][0]['fbifm'])

        # Three spatial, fbofm_init == True.
        segment = PipelineSegment((('conv3',), ('conv4',), ('conv5', 'pool3')),
                                  net, self.batch_size, self.resource)
        self.assertTrue(segment.valid)
        self.assertTrue(segment.cstr_symargs[0][0]['fbofm'])
        self.assertTrue(segment.cstr_symargs[1][0]['fbifm'])
        self.assertFalse(segment.cstr_symargs[1][0].get('fbofm', False))
        self.assertFalse(segment.cstr_symargs[2][0].get('fbifm', False))

    def test_gen_constraint_sh_mem_src(self):
        ''' gen_constraint() shared mem src. '''

        net = self.net['net3']

        segment = self._make_segment((6, 7, 8, 9), net)
        self.assertTrue(segment.valid)

        # 0 and 3 share memory source.
        for constraint, _ in segment.gen_constraint():
            self._validate_constraint(segment, constraint)

            self.assertEqual(constraint[3][0].topifm, constraint[0][0].topifm)
            self.assertTrue(constraint[3][0].topifm <= 1
                            or constraint[3][0].topofm <= 1)
            self.assertTrue(constraint[0][0].topifm <= 1
                            or constraint[0][0].topofm <= 1)

        net = self.net['net5']

        segment = self._make_segment((1, 2, 3), net)
        self.assertTrue(segment.valid)

        # 0 and 2 share memory source.
        for constraint, _ in segment.gen_constraint():
            self._validate_constraint(segment, constraint)

            # 0 constrains topofm.
            self.assertNotEqual(constraint[0][0].topofm, 0)

            # Must fully buffer ifmaps.
            self.assertEqual(constraint[2][0].topifm, 1)
            self.assertEqual(constraint[0][0].topifm, 1)

        net = self.net['net4']

        segment = self._make_segment((8, 9), net)
        self.assertTrue(segment.valid)

        # 0 and 1 share memory source.
        for constraint, _ in segment.gen_constraint():
            self._validate_constraint(segment, constraint)

            # No topofm constraint.
            self.assertEqual(constraint[0][0].topofm, 0)
            self.assertEqual(constraint[1][0].topofm, 0)

            self.assertEqual(constraint[1][0].topifm, constraint[0][0].topifm)

    def test_gen_constraint_temporal(self):
        ''' gen_constraint() temporal. '''

        for net_name in self.net:

            net = self.net[net_name]

            for segment in self._gen_all_segment(net, temporal=True):
                if not segment.valid:
                    continue

                for constraint, _ in segment.gen_constraint():
                    self._validate_constraint(segment, constraint)

    def test_gen_constraint_hints(self):
        ''' gen_constraint() pruning hints. '''

        # Use ZFNet to give the real fmap dimensions.
        net_name = 'zfnet'

        net = self.net[net_name]

        for segment in self._gen_all_segment(net):
            if not segment.valid:
                continue

            hints_set = set()
            last_hints = None

            for _, hints in segment.gen_constraint():

                self.assertTrue(all(isinstance(h, int) and h > 0
                                    for h in hints),
                                'test_gen_constraint_hints: '
                                'all hints should be positive integers only. '
                                '{}'.format(hints))

                self.assertTrue(all(
                    not all(h < ph for h, ph in zip(hints, phints))
                    for phints in hints_set),
                                'test_gen_constraint_hints: '
                                'smaller hints are generated too late.')

                if last_hints:
                    self.assertGreater(hints, last_hints,
                                       'test_gen_constraint_hints: '
                                       'hints should be generated from small '
                                       'to large.')
                last_hints = hints

    def test_gen_constraint_max_ovhd(self):
        ''' gen_constraint() with max_time_overhead. '''

        def _make_key(constraint):
            return tuple((c.topifm, c.topofm, c.topbat)
                         for c in itertools.chain.from_iterable(constraint))

        net = self.net['zfnet']

        for segment in self._gen_all_segment(net):
            if not segment.valid:
                continue

            set_all = set()
            set_1 = set()
            set_5 = set()

            for constraint, _ in segment.gen_constraint():

                timing = PipelineSegmentTiming(net, 0)
                for sp_idx, (ltpl, ctpl) in enumerate(zip(segment, constraint)):
                    for tm_idx, (l, c) in enumerate(zip(ltpl, ctpl)):
                        res = self._make_sched_res((0, sp_idx, tm_idx),
                                                   65536 // len(ltpl),
                                                   top_ti=c.topifm,
                                                   top_to=c.topofm,
                                                   top_tb=c.topbat)
                        timing.add(l, res)

                key = _make_key(constraint)

                set_all.add(key)
                if timing.time_overhead <= 0.1:
                    set_1.add(key)
                if timing.time_overhead <= 0.5:
                    set_5.add(key)

            for constraint, _ in segment.gen_constraint(max_time_overhead=0.1):
                key = _make_key(constraint)
                set_1.discard(key)

            self.assertFalse(set_1,
                             'gen_constraint with max_time_overhead: '
                             'miss generating constraints with <= 0.1 ovhd:\n'
                             '{}'.format(set_1))

            for constraint, _ in segment.gen_constraint(max_time_overhead=0.5):
                key = _make_key(constraint)
                set_5.discard(key)

            self.assertFalse(set_5,
                             'gen_constraint with max_time_overhead: '
                             'miss generating constraints with <= 0.5 ovhd:\n'
                             '{}'.format(set_5))

    def test_gen_constraint_not_opt(self):
        ''' gen_constraint() not with opt. '''

        def _validate_fully_buffered_constraint(segment, constraint):
            layer2idx = dict((l, (sp_idx, tm_idx))
                             for sp_idx, ltpl in enumerate(segment)
                             for tm_idx, l in enumerate(ltpl))
            seg_layers = set(layer2idx.keys())

            for l, c in zip(itertools.chain.from_iterable(segment),
                            itertools.chain.from_iterable(constraint)):

                if not isinstance(net[l], ConvLayer):
                    continue

                onchip_prevs = seg_layers.intersection(net.prevs(l))
                if onchip_prevs:
                    self.assertEqual(c.topifm, 1)
                    for p in onchip_prevs:
                        sp_idx, tm_idx = layer2idx[p]
                        p_c = constraint[sp_idx][tm_idx]
                        self.assertEqual(p_c.topofm, 1)

        for net_name in self.net:

            net = self.net[net_name]

            # Spatial pipelining.
            for segment in self._gen_all_segment(net, with_opt=False):
                if not segment.valid:
                    continue

                for constraint, _ in segment.gen_constraint():
                    _validate_fully_buffered_constraint(segment, constraint)

