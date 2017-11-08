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

import itertools

from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import PipelineSegment

from . import TestPipelineFixture

class TestPipelineSegment(TestPipelineFixture):
    ''' Tests for PipelineSegment. '''

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
        with self.assertRaisesRegexp(TypeError,
                                     'PipelineSegment: .*seg.*tuple.*'):
            _ = PipelineSegment([('0',), ('1', '1p')],
                                self.net['net1'], self.batch_size,
                                self.resource)

        with self.assertRaisesRegexp(TypeError,
                                     'PipelineSegment: .*seg.*sub-tuple.*'):
            _ = PipelineSegment(('0', '1', '1p'),
                                self.net['net1'], self.batch_size,
                                self.resource)

    def test_invalid_network(self):
        ''' Invalid network. '''
        with self.assertRaisesRegexp(TypeError,
                                     'PipelineSegment: .*network.*'):
            _ = PipelineSegment((('0',), ('1', '1p')),
                                self.net['net1'].input_layer(), self.batch_size,
                                self.resource)

    def test_invalid_resource(self):
        ''' Invalid resource. '''
        with self.assertRaisesRegexp(TypeError,
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
        self.assertTrue(nodes == [24, 40] or nodes == [22, 42])

        psr = self._make_segment((1, 2, 3), net)._alloc_proc()
        nodes = [nr.dim.size() for nr in psr]
        self.assertTrue(nodes == [12, 20, 32] or nodes == [10, 20, 34])

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
                    self.assertEqual(r.src_data_region().type, NodeRegion.PROC,
                                     'test_segment_allocation: '
                                     'data source should be PROC region.')

                # The data destination of all layers except for the last in the
                # segment should be local.
                for r in rlist[:-1]:
                    self.assertEqual(r.dst_data_region().type, NodeRegion.PROC,
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

    def test_allocation_temp(self):
        ''' allocation() temporal. '''

        for net in self.net.values():

            for segment in self._gen_all_segment(net, temporal=True):

                alloc = segment.allocation()
                if alloc is None:
                    continue

                self._validate_allocation(segment, alloc)

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

                for constraint, _, _ in segment.gen_constraint():
                    self._validate_constraint(segment, constraint)

                    # No top loop constraint for single-vertex segment.
                    for ctpl in constraint:
                        for c in ctpl:
                            self.assertTupleEqual(c.top_bl_t, (None,) * le.NUM)
                            self.assertEqual(c.top_bl_lpe, None)

        # Spatial pipelining.

        for net_name in self.net:

            net = self.net[net_name]

            for segment in self._gen_all_segment(net):
                if not segment.valid:
                    continue

                for constraint, _, _ in segment.gen_constraint():
                    self._validate_constraint(segment, constraint)

        # Special cases.

        net = self.net['net2']

        segment = PipelineSegment((('0', '1'), ('2', '3')), net,
                                  self.batch_size, self.resource)

        for constraint, _, _ in segment.gen_constraint():
            self._validate_constraint(segment, constraint)

    def test_gen_constraint_temporal(self):
        ''' gen_constraint() temporal. '''

        for net_name in self.net:

            net = self.net[net_name]

            for segment in self._gen_all_segment(net, temporal=True):
                if not segment.valid:
                    continue

                for constraint, _, _ in segment.gen_constraint():
                    self._validate_constraint(segment, constraint)

                    # Single spatial scheduling in temporal pipelining do not
                    # require top BAT loop.
                    for ctpl in constraint:
                        for c in ctpl:
                            self.assertIsNone(c.top_bl_t[le.BAT])
                            self.assertIsNone(c.top_bl_lpe)

    def test_gen_constraint_opt_step(self):
        ''' gen_constraint() pruning info opt_step. '''

        for net_name in self.net:

            if not net_name.startswith('net') and net_name != 'zfnet':
                # Use ZFNet to give the real fmap dimensions.
                continue
            net = self.net[net_name]

            for segment in self._gen_all_segment(net):
                if not segment.valid:
                    continue

                ostep_set_ref = None
                ostep_set = set()
                last_fmap_tpart = 1

                for constraint, ostep, _ in segment.gen_constraint():

                    fmap_tpart = constraint[0][0].fmap_tpart

                    if ostep:
                        # Comes a new opt step. Close the current one.

                        # Update fmap tpart.
                        self.assertGreater(fmap_tpart, last_fmap_tpart,
                                           'test_gen_constraint_opt_step: '
                                           'fmap tpart is not monotonically '
                                           'increasing across opt steps. '
                                           '{} -> {}.'
                                           .format(last_fmap_tpart, fmap_tpart))
                        last_fmap_tpart = fmap_tpart

                        # Compare to ref.
                        ostep_set_2 = set()
                        for cstr in ostep_set:
                            # Replace fmap tpart and top tb.
                            # Different fmap tpart may lead to different top tb.
                            cstr_2 = tuple(tuple(c._replace(
                                fmap_tpart=0,
                                top_bl_t=(c.top_bl_t[:le.BAT]
                                          + (0,)
                                          + c.top_bl_t[le.BAT + 1:]))
                                                 for c in ctpl)
                                           for ctpl in cstr)
                            ostep_set_2.add(cstr_2)
                        # Larger fmap tpart value may introduce more valid
                        # constraints.
                        self.assertTrue(
                            ostep_set_ref is None
                            or ostep_set_ref.issubset(ostep_set_2),
                            'test_gen_constraint_opt_step: constraints '
                            'differ across opt steps. '
                            'Network {}, segment {}.'
                            .format(net_name, segment))
                        ostep_set_ref = ostep_set_2

                        ostep_set.clear()

                    ostep_set.add(constraint)
                    self.assertEqual(fmap_tpart, last_fmap_tpart,
                                     'test_gen_constraint_opt_step: fmap tpart '
                                     'is not constant within an opt step. '
                                     '{} != {}.'
                                     .format(fmap_tpart, last_fmap_tpart))

    def test_gen_constraint_ff_end(self):
        ''' gen_constraint() pruning info ff_end. '''

        for net_name in self.net:

            if not net_name.startswith('net') and net_name != 'zfnet':
                # Use ZFNet to give the real fmap dimensions.
                continue
            net = self.net[net_name]

            for segment in self._gen_all_segment(net):
                if not segment.valid:
                    continue

                last_top_tb = 0
                last_cstr = None

                for constraint, ostep, ff_end in segment.gen_constraint():

                    top_tb = constraint[0][0].top_bl_t[le.BAT]

                    if ostep:
                        self.assertTrue(ff_end,
                                        'test_gen_constraint_ff_end: '
                                        'ff_end must be True when a new '
                                        'opt step starts.')

                    if last_top_tb is None:
                        self.assertTrue(ff_end,
                                        'test_gen_constraint_ff_end: '
                                        'ff_end must be True when previous '
                                        'top tb is None.')

                    # Replace top tb.
                    cstr = tuple(tuple(c._replace(
                        top_bl_t=(c.top_bl_t[:le.BAT]
                                  + (0,)
                                  + c.top_bl_t[le.BAT + 1:]))
                                       for c in ctpl) for ctpl in constraint)

                    if last_cstr and not ff_end:
                        self.assertEqual(cstr, last_cstr,
                                         'test_gen_constraint_ff_end: '
                                         'with False ff_end, constraints must '
                                         'be the same except for top tb. '
                                         'current {}; last {}.'
                                         .format(cstr, last_cstr))
                        self.assertGreater(top_tb, last_top_tb,
                                           'test_gen_constraint_ff_end: '
                                           'with False ff_end, top tb must '
                                           'increase. current {}; last {}.'
                                           .format(top_tb, last_top_tb))

                    last_cstr = cstr
                    last_top_tb = top_tb

    def test_gen_constraint_fmap_tpart(self):
        ''' gen_constraint() valid fmap_tpart. '''

        net = self.net['net6']

        segment = self._make_segment((0, 1, 2, 3), net, max_util_drop=1.)
        self.assertTrue(segment.valid)

        fmap_tpart_set = set()
        for constraint, _, _ in segment.gen_constraint():
            fmap_tpart_set.add(constraint[0][0].fmap_tpart)

        # The candidates will be 1 to 6. 5 is not valid due to dividability,
        # and 6 is not valid due to pyramid enlargement.
        self.assertNotIn(5, fmap_tpart_set)
        self.assertNotIn(6, fmap_tpart_set)

