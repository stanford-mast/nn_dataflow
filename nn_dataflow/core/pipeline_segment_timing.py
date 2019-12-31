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

from collections import namedtuple, OrderedDict

from . import loop_enum as le
from .loop_blocking_scheme import LoopBlockingScheme
from .layer import ConvLayer
from .network import Network

class PipelineSegmentTiming():
    ''' Timing information of a pipeline segment. '''

    # Each layer timing info is a tuple:
    # - time: the total time.
    # - node_time: the total time on node processing.
    # - dram_time: the total time on DRAM access.
    # - num_nodes: the number of processing nodes.
    # - ngrp: the OFM group number.
    # - ts_xb: when to start.
    # - td_xb: when the first BAT group of this and all prev layers is done.
    # Time is stored by multiplying the lazily updated BAT group number (_xb).
    # Notice (td - ts) may be greater than (time), because fused layers can
    # have an earlier start time, but done time is sequentially accumulated.
    LayerTiming = namedtuple('LayerTiming', ['time', 'node_time', 'dram_time',
                                             'num_nodes', 'ngrp',
                                             'ts_xb', 'td_xb'])

    def __init__(self, network, seg_idx):

        if not isinstance(network, Network):
            raise TypeError('PipelineSegmentTiming: network must be a '
                            'Network instance.')
        self.network = network

        # Scheduling sequence number.
        self.seg_idx = seg_idx
        self.last_sched_seq = None

        # Time properties.
        # The time on DRAM accesses.
        self.dram_time = 0
        # The time on node processing.
        self.node_time = 0
        # The critical (longest) spatial scheduling time.
        self.critical_time = 0

        # Mapping from layer name to spatial and temporal indices.
        self.layer2idx = OrderedDict()

        # The number of groups of which BAT are sequentially processed, i.e.,
        # the degree of batch pipelining, shared by all layers in the segment.
        # Lazily updated.
        self.bat_ngrp = None

        # Timing of each layer, indexed by spatial and temporal indices.
        self.timing_list = []

    @property
    def time(self):
        ''' The total time of the end-to-end segment processing. '''
        return max(self.node_time, self.dram_time)

    @property
    def time_overhead(self):
        '''
        The time overhead as a percentage, to process layers in segment
        compared to processing layers individually.
        '''
        total_num_nodes = sum(tlist[0].num_nodes
                              for tlist in self.timing_list)
        # Sum up the max of scaled node time and DRAM time.
        time_indv = sum(max(1. * timing.node_time * timing.num_nodes
                            / total_num_nodes,
                            timing.dram_time)
                        for tlist in self.timing_list
                        for timing in tlist)
        return (self.time - time_indv) / time_indv

    def add(self, layer_name, sched_result):
        ''' Add the SchedulingResult of a new layer. '''

        sched_seq = sched_result.sched_seq

        if sched_seq[0] != self.seg_idx:
            raise ValueError('PipelineSegmentTiming: sched_seq {} does not '
                             'belong to segment {}.'
                             .format(sched_seq, self.seg_idx))

        if sched_seq == self._sched_seq_incr(1):
            # New spatial scheduling.
            self.timing_list.append([])
        elif sched_seq == self._sched_seq_incr(2):
            # New temporal scheduling.
            pass
        else:
            raise ValueError('PipelineSegmentTiming: sched_seq {} cannot '
                             'follow {}'
                             .format(sched_seq, self.last_sched_seq))
        self.last_sched_seq = sched_seq

        if layer_name in self.layer2idx:
            raise ValueError('PipelineSegmentTiming: layer {} already in '
                             'segment, old sched_seq {}, new sched_seq {}.'
                             .format(layer_name, self.layer2idx[layer_name],
                                     sched_seq[1:]))
        self.layer2idx[layer_name] = sched_seq[1:]

        # Add layer timing.

        timing = self._make_layer_timing(layer_name, sched_result)
        assert not self.timing_list[-1] \
                or timing.num_nodes == self.timing_list[-1][-1].num_nodes
        self.timing_list[-1].append(timing)
        assert self.last_sched_seq[1] + 1 == len(self.timing_list)
        assert self.last_sched_seq[2] + 1 == len(self.timing_list[-1])

        # Update time.

        # Critical time, as the longest of all spatial scheduling.
        assert all(sum(timing.time for timing in tlist)
                   <= tlist[-1].td_xb - tlist[0].ts_xb
                   for tlist in self.timing_list)
        self.critical_time = max(tlist[-1].td_xb - tlist[0].ts_xb
                                 for tlist in self.timing_list)

        # DRAM time.
        # Each layer DRAM time is calculated using the layer accesses and the
        # maximum bandwidth. Accumulating the accesses is accumulating the
        # time.
        self.dram_time += sched_result.total_dram_time

        # Node time, as the max of end time of the last BAT group.
        # The interval between BAT groups is determined by the critical time of
        # one BAT group.
        self.node_time = max((tlist[-1].td_xb
                              + self.critical_time * (self.bat_ngrp - 1))
                             // self.bat_ngrp
                             for tlist in self.timing_list)
        assert self.node_time >= self.critical_time

    def _sched_seq_incr(self, pos):
        ''' Get the next sched seq incremented at the given position. '''
        if not self.last_sched_seq:
            return (self.seg_idx, 0, 0)
        assert len(self.last_sched_seq) == 3
        return self.last_sched_seq[:pos] + (self.last_sched_seq[pos] + 1,) \
                + (0,) * (2 - pos)

    def _make_layer_timing(self, layer_name, sched_result):
        ''' Construct and return the layer timing. '''
        # Top-level ordered loops, from outermost to innermost.
        ord_loops = LoopBlockingScheme.ordered_loops(
            sched_result.scheme['tvals'][0], sched_result.scheme['orders'][0])

        # Top loop blocking factors.
        top_ts = [1] * le.NUM
        if ord_loops and ord_loops[0][0] == le.BAT:
            top_ts[le.BAT] = ord_loops.pop(0)[1]
        if ord_loops:
            lpe, t = ord_loops.pop(0)
            assert lpe in (le.IFM, le.OFM)
            top_ts[lpe] = t

        # Lazily update BAT group number.
        if not self.bat_ngrp:
            self.bat_ngrp = top_ts[le.BAT]
        elif self.bat_ngrp != top_ts[le.BAT]:
            # Unmatched.
            self.bat_ngrp = 1

        # IFM/OFM group number.
        ifm_ngrp, ofm_ngrp = top_ts[le.IFM], top_ts[le.OFM]

        # Time on node processing and DRAM access.
        node_time = sched_result.total_node_time
        dram_time = sched_result.total_dram_time
        # Number of nodes.
        num_nodes = sched_result.num_nodes

        # Calculate timing.
        sp_idx, tm_idx = self.layer2idx[layer_name]
        is_conv = isinstance(self.network[layer_name], ConvLayer)
        time = sched_result.total_time
        ts_xb = 0
        td_xb = 0
        for p in self.network.prevs(layer_name):
            if p not in self.layer2idx:
                # Off-chip source.
                continue
            # On-chip source.
            p_sp_idx, p_tm_idx = self.layer2idx[p]
            p_timing = self.timing_list[p_sp_idx][p_tm_idx]
            if p_sp_idx == sp_idx:
                assert p_tm_idx == tm_idx - 1
                # Same spatial scheduling.
                if not is_conv and ofm_ngrp == p_timing.ngrp:
                    # Fused.
                    start = p_timing.ts_xb + p_timing.time // p_timing.ngrp
                else:
                    # Not fused.
                    start = p_timing.td_xb
                # Also constrain the done time.
                td_xb = p_timing.td_xb + time
            else:
                assert p_sp_idx < sp_idx
                assert p_tm_idx == len(self.timing_list[p_sp_idx]) - 1
                # Previous spatial scheduling.
                if (ifm_ngrp if is_conv else ofm_ngrp) == p_timing.ngrp:
                    # I/OFM group forwarding.
                    start = p_timing.ts_xb + p_timing.time // p_timing.ngrp
                else:
                    # All I/OFM double buffering.
                    start = p_timing.td_xb
            ts_xb = max(ts_xb, start)
        td_xb = max(td_xb, ts_xb + time)

        return PipelineSegmentTiming.LayerTiming(
            time=time, node_time=node_time, dram_time=dram_time,
            num_nodes=num_nodes, ngrp=ofm_ngrp, ts_xb=ts_xb, td_xb=td_xb)

