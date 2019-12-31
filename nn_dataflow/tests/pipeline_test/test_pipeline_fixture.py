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
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer
from nn_dataflow.core import InterLayerPipeline
from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import Network
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import PipelineSegment
from nn_dataflow.core import Resource
from nn_dataflow.core import SchedulingConstraint
from nn_dataflow.core import SchedulingResult

from nn_dataflow.nns import import_network, all_networks

class TestPipelineFixture(unittest.TestCase):
    ''' Base fixture class for layer pipeline tests. '''

    def setUp(self):

        self.net = {}

        net = Network('net1')
        # Linear.
        net.set_input_layer(InputLayer(10, 1))
        net.add('0', FCLayer(10, 20))
        net.add('1', FCLayer(20, 30))
        net.add('1p', PoolingLayer(30, 1, 1))
        net.add('2', FCLayer(30, 40))
        net.add('3', FCLayer(40, 50))
        self.net[net.net_name] = net

        net = Network('net2')
        # Long linear.
        net.set_input_layer(InputLayer(1, 1))
        for idx in range(16):
            net.add(str(idx), FCLayer(1, 1))
        self.net[net.net_name] = net

        net = Network('net3')
        # Fork.
        # /0-2\   /6- 7- 8\
        #   x  4-5         12
        # \1-3/   \9-10-11/
        net.set_input_layer(InputLayer(1, 1))
        net.add('0', FCLayer(1, 1), prevs=net.INPUT_LAYER_KEY)
        net.add('1', FCLayer(1, 1), prevs=net.INPUT_LAYER_KEY)
        net.add('2', FCLayer(2, 1), prevs=('0', '1'))
        net.add('2p', PoolingLayer(1, 1, 1))
        net.add('3', FCLayer(2, 1), prevs=('0', '1'))
        net.add('4', FCLayer(2, 1), prevs=('2p', '3'))
        net.add('5', FCLayer(1, 1))
        net.add('5p', PoolingLayer(1, 1, 1))
        net.add('6', FCLayer(1, 1), prevs='5p')
        net.add('7', FCLayer(1, 1))
        net.add('8', FCLayer(1, 1))
        net.add('9', FCLayer(1, 1), prevs='5p')
        net.add('10', FCLayer(1, 1))
        net.add('11', FCLayer(1, 1))
        net.add('12', FCLayer(2, 1), prevs=('8', '11'))
        self.net[net.net_name] = net

        net = Network('net4')
        # Complex fork.
        #          /5       \
        # 0-1-2-3-4-6-7-8-10-14
        #              \9/
        #          \11-12   /
        #          \13      /
        net.set_input_layer(InputLayer(1, 1))
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
        net.add('10p', PoolingLayer(2, 1, 1), prevs=('8', '10'))
        net.add('11', PoolingLayer(1, 1, 1), prevs='4')
        net.add('12', FCLayer(1, 1))
        net.add('13', PoolingLayer(1, 1, 1), prevs='4')
        net.add('14', FCLayer(5, 1), prevs=('5', '10p', '12', '13'))
        self.net[net.net_name] = net

        net = Network('net5')
        # Corner cases.
        #  ----\
        # //1-2\ 7-8\
        # 0-3-4-x   10-11-12
        #  \ \5/ 9 /  \__/
        #   6--/
        net.set_input_layer(InputLayer(1, 1))
        net.add('0', FCLayer(1, 1))
        net.add('1', FCLayer(1, 1), prevs='0')
        net.add('2', FCLayer(1, 1))
        net.add('3', FCLayer(1, 1), prevs='0')
        net.add('4', FCLayer(1, 1), prevs='3')
        net.add('5', FCLayer(1, 1), prevs='3')
        net.add('6', FCLayer(1, 1), prevs='0')
        net.add('7', FCLayer(5, 1), prevs=('0', '2', '4', '5', '6'))
        net.add('8', FCLayer(1, 1))
        net.add('9', FCLayer(5, 1), prevs=('0', '2', '4', '5', '6'))
        net.add('10', FCLayer(2, 1), prevs=('8', '9'))
        net.add('11', FCLayer(1, 1))
        net.add('12', FCLayer(2, 1), prevs=('10', '11'))
        self.net[net.net_name] = net

        net = Network('net6')
        # Fmap sizes.
        net.set_input_layer(InputLayer(1, 24))
        net.add('0', ConvLayer(1, 1, 24, 3))
        net.add('1', ConvLayer(1, 1, 12, 3, strd=2))
        net.add('1p', PoolingLayer(1, 6, 2))
        net.add('2', ConvLayer(1, 1, 6, 3))
        net.add('3', ConvLayer(1, 1, 6, 3, strd=4), prevs=('0'))
        self.net[net.net_name] = net

        net = Network('net7')
        # Topological order: see a visited vertex again.
        #  /---
        # 0-1-\\
        #  \2--2p
        net.set_input_layer(InputLayer(1, 1))
        net.add('0', FCLayer(1, 1))
        net.add('1', FCLayer(1, 1), prevs='0')
        net.add('2', FCLayer(1, 1), prevs='0')
        net.add('2p', PoolingLayer(3, 1, 1), prevs=('0', '1', '2'))
        self.net[net.net_name] = net

        net = Network('net8')
        # Forward to the middle.
        #    /-\
        # 0-1-2-2p-4-4p
        #  \-3------/
        net.set_input_layer(InputLayer(1, 1))
        net.add('0', FCLayer(1, 1))
        net.add('1', FCLayer(1, 1), prevs='0')
        net.add('2', FCLayer(1, 1), prevs='1')
        net.add('2p', PoolingLayer(2, 1, 1), prevs=('1', '2'))
        net.add('3', FCLayer(1, 1), prevs='0')
        net.add('4', FCLayer(2, 1), prevs='2p')
        net.add('4p', PoolingLayer(2, 1, 1), prevs=('3', '4'))
        self.net[net.net_name] = net

        net = Network('net9')
        # Previous layers include input and others.
        net.set_input_layer(InputLayer(1, 1))
        net.add('0', FCLayer(1, 1))
        net.add('1', FCLayer(2, 1), prevs=(net.INPUT_LAYER_KEY, '0'))
        self.net[net.net_name] = net

        # Real networks.
        for net_name in all_networks():
            self.net[net_name] = import_network(net_name)

        self.batch_size = 16

        self.resource = Resource(
            proc_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(8, 8),
                                   type=NodeRegion.PROC),
            dram_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(8, 8),
                                   type=NodeRegion.DRAM),
            src_data_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(8, 4),
                                       type=NodeRegion.DRAM),
            dst_data_region=NodeRegion(origin=PhyDim2(0, 4), dim=PhyDim2(8, 4),
                                       type=NodeRegion.DRAM),
            dim_array=PhyDim2(16, 16), size_gbuf=65536, size_regf=64,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)

        part = PartitionScheme(order=range(pe.NUM), pdims=[(1, 1)] * pe.NUM)
        self.ofmap_layout = DataLayout(
            frngs=(FmapRange((0, 0, 0, 0), (2, 4, 16, 16)),),
            regions=(NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                type=NodeRegion.DRAM),),
            parts=(part,))


    def _make_ilp(self, network):
        ''' Make an InterLayerPipeline instance. '''
        return InterLayerPipeline(network, self.batch_size, self.resource)

    def _make_segment(self, vseg, network, temporal=False, max_util_drop=None,
                      with_opt=True):
        ''' Convert vertex segment to (layer) segment. '''
        kwargs = {}
        if max_util_drop is not None:
            kwargs['max_util_drop'] = max_util_drop
        if not with_opt:
            kwargs['with_opt'] = False
        ilp = self._make_ilp(network)
        seg = tuple(ilp.dag_vertex_list[vidx] for vidx in vseg)
        if temporal:
            seg = (sum(seg, tuple()),)
        return PipelineSegment(seg, ilp.network, ilp.batch_size, ilp.resource,
                               **kwargs)

    def _make_sched_res(self, sched_seq, time, top_ti=1, top_to=1, top_tb=1,
                        top_ord=range(le.NUM), dram_time=0, num_nodes=4):
        scheme = OrderedDict()
        scheme['cost'] = 1.234 + 9.876
        scheme['time'] = max(time, dram_time)
        scheme['num_nodes'] = num_nodes
        scheme['proc_time'] = time
        scheme['bus_time'] = 0
        scheme['dram_time'] = dram_time
        scheme['ti'] = [top_ti, 1, 1]
        scheme['to'] = [top_to, 1, 1]
        scheme['tb'] = [top_tb, 1, 1]
        scheme['tvals'] = [[top_ti, top_to, top_tb], [1] * 3, [1] * 3]
        scheme['orders'] = [top_ord, range(le.NUM), range(le.NUM)]
        return SchedulingResult(scheme=scheme,
                                ofmap_layout=self.ofmap_layout,
                                sched_seq=sched_seq)

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
            segment = self._make_segment(vseg, network, **kwargs)
            if len(segment) == 1 and len(segment[0]) == 1:
                continue
            yield segment

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
            for n in proc_region.iter_node():
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
                prev_layers = segment.network.prevs(l)

                for pl in prev_layers:
                    if pl not in data_regions:
                        # Previous layer is not on-chip, from memory.
                        # Try find a layer responsible to fetch shared mem src.
                        try:
                            sh_sp_idx = next((i for i in range(len(allocation))
                                              if allocation[i][0].proc_region
                                              == r.src_data_region))
                        except StopIteration:
                            # No shared mem src.
                            self.assertEqual(
                                r.src_data_region,
                                self.resource.src_data_region,
                                '_validate_allocation: layer {}\'s prev {} '
                                'is not on-chip, should be from {}, but {}.'
                                .format(l, pl, self.resource.src_data_region,
                                        r.src_data_region))
                        else:
                            # There exists shared mem src.
                            sh_l = segment[sh_sp_idx][0]
                            self.assertEqual(segment.network.prevs(l),
                                             segment.network.prevs(sh_l),
                                             '_validate_allocation: layer {} '
                                             'expects on-chip mem src sharing '
                                             'with {}, but prevs differ.'
                                             .format(l, sh_l))
                    elif data_regions[pl] != r.proc_region:
                        # Previous layer is on-chip and not local.
                        self.assertEqual(
                            r.src_data_region, data_regions[pl],
                            '_validate_allocation: layer {}\'s prev {} '
                            'is on-chip, should be from {}, but {}.'
                            .format(l, pl, data_regions[pl],
                                    r.src_data_region))

                # Update data based on destination.
                # Local or store back to memory. Both will be available on-chip.
                self.assertTrue(r.dst_data_region == r.proc_region
                                or r.dst_data_region
                                == self.resource.dst_data_region,
                                '_validate_allocation: data can only '
                                'be local or storing back to mem.')
                # Overwrite.
                local_node_set = set(r.proc_region.iter_node())
                data_regions = {pl: data_regions[pl] for pl in data_regions
                                if local_node_set.isdisjoint(
                                    data_regions[pl].iter_node())}
                data_regions[l] = r.proc_region

    def _validate_constraint(self, segment, constraint):
        ''' Validate segment scheduling constraint. '''
        # pylint: disable=too-many-branches

        # Match segment.
        self.assertEqual(len(constraint), len(segment))
        for ltpl, ctpl in zip(segment, constraint):
            self.assertEqual(len(ctpl), len(ltpl))
            self.assertTrue(all(isinstance(c, SchedulingConstraint)
                                for c in ctpl))

        # Same top tb.
        top_tb = constraint[0][0].topbat
        self.assertTrue(all(c.topbat == top_tb
                            for ctpl in constraint for c in ctpl))

        # Top tb is a factor of batch size.
        if top_tb:
            self.assertEqual((segment.batch_size) % top_tb, 0)

        # Data availability.

        seg_layers = set(l for ltpl in segment for l in ltpl)

        class OutAccPat():
            ''' Output data access pattern types. '''
            # pylint: disable=too-few-public-methods
            ANY = 0   # can access in any way
            DBF = -1  # must double-buffer
            # SEQ: use any positive value to represent sequential access with
            # certain number of groups.

        # Available data in each spatial subregions. Each is represented by a
        # tuple of layer name and its output data access pattern.
        avail_data = [(None, OutAccPat.ANY) for _ in segment]

        # Get groups of layers sharing the same memory source.
        prevs2layers = {}
        for ltpl in segment:
            l = ltpl[0]
            prevs2layers.setdefault(segment.network.prevs(l), []).append(l)
        sh_mem_src_groups = [ls for ps, ls in prevs2layers.items()
                             if not seg_layers.intersection(ps) and len(ls) > 1]
        sh_mem_src_topifms = [None] * len(sh_mem_src_groups)

        # Whether to defer fully buffering output.
        fb_out = False
        fb_out_conv = None

        for sp_idx, (ltpl, ctpl) in enumerate(zip(segment, constraint)):

            self.assertFalse(fb_out,
                             '_validate_constraint: deferring fully buffering '
                             'from {} should not cross spatial scheduling {}.'
                             .format(fb_out_conv, sp_idx - 1))

            for tm_idx, (layer, cstr) in enumerate(zip(ltpl, ctpl)):

                # Source data and their access patterns.
                prev_layers = segment.network.prevs(layer)
                prev_oaps = []
                for pl in prev_layers:
                    if pl not in seg_layers:
                        # Off-chip sources.
                        poap = OutAccPat.ANY
                    elif pl in ltpl:
                        # On-chip and local.
                        self.assertEqual(avail_data[sp_idx][0], pl,
                                         '_validate_constraint: layer {} ({}) '
                                         'local source data {} not available, '
                                         'maybe not the immediate previous.'
                                         .format(layer, (sp_idx, tm_idx), pl))
                        poap = avail_data[sp_idx][1]
                    else:
                        # On-chip and neighbor.
                        poap = next((avail_data[p_sp_idx][1]
                                     for p_sp_idx in range(sp_idx)
                                     if avail_data[p_sp_idx][0] == pl),
                                    None)
                        self.assertFalse(poap is None,
                                         '_validate_constraint: layer {} ({}) '
                                         'neighbor source data {} not '
                                         'available on-chip.'
                                         .format(layer, (sp_idx, tm_idx), pl))
                    prev_oaps.append(poap)
                # Only buffer input if having source on-chip.
                has_src = not seg_layers.isdisjoint(prev_layers)

                # The single SEQ source.
                seq = None
                # str is greater than all numbers, see
                # https://docs.python.org/2/library/stdtypes.html#comparisons
                seq_prev_oaps = [poap for poap in prev_oaps \
                                 if isinstance(poap, str) or \
                                    (isinstance(poap, int) and poap > 0)]
                if seq_prev_oaps:
                    self.assertEqual(len(seq_prev_oaps), 1,
                                     '_validate_constraint: layer {} ({}) '
                                     'has multiple SEQ input.'
                                     '\nsrcs: {}, oaps: {}'
                                     .format(layer, (sp_idx, tm_idx),
                                             prev_layers, prev_oaps))
                    seq = seq_prev_oaps[0]

                # Destination data.
                # Only buffer output if having destination on-chip.
                next_layers = segment.network.nexts(layer)
                has_dst = not seg_layers.isdisjoint(next_layers)

                # Validation.

                for g_idx, group in enumerate(sh_mem_src_groups):
                    if layer in group:
                        if sh_mem_src_topifms[g_idx] is None:
                            sh_mem_src_topifms[g_idx] = cstr.topifm
                        self.assertEqual(sh_mem_src_topifms[g_idx], cstr.topifm,
                                         '_validate_constraint: layer {} ({}) '
                                         'share memory source with {}, but has '
                                         'mismatched topifm {} with {}.'
                                         .format(layer, (sp_idx, tm_idx),
                                                 group, cstr.topifm,
                                                 sh_mem_src_topifms[g_idx]))
                        break
                else:
                    if not has_src:
                        self.assertEqual(cstr.topifm, 0,
                                         '_validate_constraint: layer {} ({}) '
                                         'should not constrain input as it '
                                         'does not have on-chip sources.'
                                         .format(layer, (sp_idx, tm_idx)))

                if isinstance(segment.network[layer], ConvLayer):

                    self.assertFalse(fb_out,
                                     '_validate_constraint: deferring fully '
                                     'buffering from {} has not been realized.'
                                     .format(fb_out_conv))

                    if any(pl in ltpl for pl in prev_layers):
                        # Local source.
                        lcl_poap = avail_data[sp_idx][1]
                        self.assertIn(lcl_poap, (OutAccPat.DBF, OutAccPat.ANY),
                                      '_validate_constraint: layer {} ({}) '
                                      'local source data {} must fully '
                                      'buffer output.'
                                      .format(layer, (sp_idx, tm_idx),
                                              lcl_poap))

                    # DBF source.
                    if OutAccPat.DBF in prev_oaps:
                        # Must fully buffer CONV input.
                        self.assertEqual(cstr.topifm, 1,
                                         '_validate_constraint: layer {} ({}) '
                                         'input is not fully buffered but has '
                                         'DBF source.\nsrcs: {}, oaps: {}'
                                         '\n{}'
                                         .format(layer, (sp_idx, tm_idx),
                                                 prev_layers, prev_oaps,
                                                 cstr))

                    # SEQ source.
                    if seq and has_dst:
                        # Cannot be lazily updated.
                        self.assertNotIsInstance(
                            seq, str,
                            '_validate_constraint: CONV layer {} ({}) cannot '
                            'use lazy update (from {})'
                            .format(layer, (sp_idx, tm_idx), seq))
                        # Must match SEQ.
                        self.assertEqual(cstr.topifm, seq,
                                         '_validate_constraint: layer {} ({}) '
                                         'input groups ({}) and its SEQ src '
                                         'output groups ({}) are mismatched.'
                                         '\nsrcs: {}, oaps: {}'
                                         .format(layer, (sp_idx, tm_idx),
                                                 cstr.topifm, seq,
                                                 prev_layers, prev_oaps))
                        # Also must fully buffer CONV output.
                        self.assertEqual(cstr.topofm, 1,
                                         '_validate_constraint: layer {} ({}) '
                                         'output is not fully buffered but has '
                                         'SEQ source.\nsrcs: {}, oaps: {}'
                                         .format(layer, (sp_idx, tm_idx),
                                                 prev_layers, prev_oaps))
                        # Deferred apply to the last layer in the group.
                        fb_out = True
                        fb_out_conv = layer

                    oap = None
                    if cstr.topofm == 1:
                        if cstr.topifm == 1:
                            # Fully buffer both, can access output in any way.
                            # This is fine as we require to buffer either input
                            # or output for CONV (see below).
                            oap = OutAccPat.ANY
                        else:
                            oap = OutAccPat.DBF
                    elif has_dst and cstr.topofm > 0:
                        oap = cstr.topofm
                        if has_src:
                            self.assertEqual(cstr.topifm, 1,
                                             '_validate_constraint: layer {} '
                                             '({}) has on-chip src and dst '
                                             'but neither input nor output '
                                             'are fully buffered.\ncstr: {}.'
                                             .format(layer, (sp_idx, tm_idx),
                                                     cstr))
                    elif has_dst:
                        # Lazy update, record layer name as seq.
                        oap = layer

                else:

                    # SEQ source.
                    if seq and has_dst:
                        # Must match SEQ, or fully buffer output.
                        self.assertTrue(cstr.topofm == seq or cstr.topofm == 1
                                        or seq in cstr.update_dict,
                                        '_validate_constraint: layer {} ({}) '
                                        'output is not fully buffered, and '
                                        'groups ({}) and its SEQ src output '
                                        'groups ({}) are mismatched, and '
                                        'lazy update is not used.'
                                        '\nsrcs: {}, oaps: {}'
                                        .format(layer, (sp_idx, tm_idx),
                                                cstr.topofm, seq,
                                                prev_layers, prev_oaps))

                    if cstr.topofm == 1:
                        # Fully buffer output.
                        oap = OutAccPat.DBF
                    elif isinstance(seq, str):
                        # Lazy update.
                        oap = seq
                    else:
                        # SEQ output.
                        oap = cstr.topofm

                # Realize deferred fully buffering output.
                if cstr.topofm == 1:
                    fb_out = False  # reset

                # Overwrite the previous temporal scheduling.
                avail_data[sp_idx] = (layer, oap)

