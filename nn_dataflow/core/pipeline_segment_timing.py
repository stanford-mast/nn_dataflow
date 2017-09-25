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

from collections import namedtuple, OrderedDict
import itertools

from . import loop_enum as le
from .. import util
from .layer import ConvLayer
from .network import Network

class PipelineSegmentTiming(object):
    ''' Timing information of a pipeline segment. '''

    # Timing info of a layer in the segment, including
    # - the total time of the layer.
    # - whether the layer is fused with its previous layer.
    # - the number of groups of which IFM are sequentially processed.
    # - the number of groups of which OFM are sequentially processed.
    LayerTiming = namedtuple('LayerTiming', ['time', 'fused',
                                             'ifm_ngrp', 'ofm_ngrp'])

    def __init__(self, network, seg_idx):

        if not isinstance(network, Network):
            raise TypeError('PipelineSegmentTiming: network must be a '
                            'Network instance.')
        self.network = network

        # Scheduling sequence number.
        self.seg_idx = seg_idx
        self.last_sched_seq = None

        # Mapping from layer name to spatial and temporal indices.
        self.layer2idx = OrderedDict()

        # Nested list for layer timing info, indexed by spatial and temporal
        # indices.
        self.timing_list = []

        # The number of groups of which BAT are sequentially processed, shared
        # by all layers in the segment. An individual layer may have a larger
        # top BAT loop factor, but must be divided by this shared factor.
        self.bat_ngrp = None

        # Cached time and critical time. Lazily calculated since the number of
        # BAT group may change.
        self.cached_time = None
        self.cached_crit_time = None

    def time(self):
        ''' The total time of the end-to-end segment schedule. '''
        self._calc_time()
        assert self.cached_time is not None
        return self.cached_time

    def critical_time(self):
        ''' The critical spatial scheduling pipeline stage time. '''
        self._calc_time()
        assert self.cached_crit_time is not None
        return self.cached_crit_time

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

        # Loop blocking scheme.
        # Ordered loops from outer to inner with LoopEnum and blocking factor.
        # Filter out trivial loops.
        ord_loops = []
        for ti, to, tb, bl_ord in zip(sched_result.dict_loop['ti'],
                                      sched_result.dict_loop['to'],
                                      sched_result.dict_loop['tb'],
                                      sched_result.dict_loop['orders']):
            bl_t = [None] * le.NUM
            bl_t[le.IFM] = ti
            bl_t[le.OFM] = to
            bl_t[le.BAT] = tb
            ord_lpe = sorted(range(le.NUM),
                             key=lambda lpe, o=bl_ord: o[lpe], reverse=True)
            ord_loops += [(lpe, bl_t[lpe]) for lpe in ord_lpe if bl_t[lpe] > 1]

        # Update the BAT groups.
        bat_ngrp = 1
        while ord_loops:
            if ord_loops[0][0] == le.BAT:
                bat_ngrp = ord_loops[0][1]
                ord_loops.pop(0)
            else:
                break
        if not self.bat_ngrp:
            self.bat_ngrp = bat_ngrp
        else:
            self.bat_ngrp = util.gcd(self.bat_ngrp, bat_ngrp)

        # Get the IFM/OFM groups.
        ifm_ngrp = util.prod(
            lp[1] for lp in itertools.takewhile(lambda lp: lp[0] == le.IFM,
                                                ord_loops))
        ofm_ngrp = util.prod(
            lp[1] for lp in itertools.takewhile(lambda lp: lp[0] == le.OFM,
                                                ord_loops))

        # Fused. Only fuse non-CONV layers and not-the-first-temporal layers.
        fused = not isinstance(self.network[layer_name], ConvLayer) \
                and sched_seq[-1] > 0

        # Construct layer timing info.
        timing = PipelineSegmentTiming.LayerTiming(
            time=sched_result.total_time, fused=fused,
            ifm_ngrp=ifm_ngrp, ofm_ngrp=ofm_ngrp)

        # Append.
        self.timing_list[-1].append(timing)
        assert self.last_sched_seq[1] + 1 == len(self.timing_list)
        assert self.last_sched_seq[2] + 1 == len(self.timing_list[-1])

        # Invalidate cached results.
        self.cached_time = None
        self.cached_crit_time = None

    def _sched_seq_incr(self, pos):
        ''' Get the next sched seq incremented at the given position. '''
        if not self.last_sched_seq:
            return (self.seg_idx, 0, 0)
        assert len(self.last_sched_seq) == 3
        return self.last_sched_seq[:pos] + (self.last_sched_seq[pos] + 1,) \
                + (0,) * (2 - pos)

    def _calc_time(self):
        ''' Calculate the time and critical time. '''

        if self.cached_time is not None and self.cached_crit_time is not None:
            return

        # Start time of each layer in the segment. Only for added layers.
        start_list = []
        for layer_name in self.layer2idx:
            self._calc_start_time(layer_name, start_list)

        # Critical stage time, as the max of all spatial scheduling.
        self.cached_crit_time = max(sum(timing.time for timing in tlist)
                                    for tlist in self.timing_list)

        # Total time, as the max of end time of the last BAT group.
        # The interval between BAT groups is determined by the critical stage
        # time of one BAT group.
        self.cached_time = max(start + timing.time // self.bat_ngrp
                               + self.cached_crit_time // self.bat_ngrp
                               * (self.bat_ngrp - 1)
                               for slist, tlist
                               in zip(start_list, self.timing_list)
                               for start, timing in zip(slist, tlist))

        assert self.cached_time >= self.cached_crit_time

    def _calc_start_time(self, layer_name, start_list):
        '''
        Calculate the start time of a layer, with the start time of previous
        layers given as a nested list indexed by spatial and temporal indices.
        Append the start time to the nested list.
        '''

        layer_idx = self.layer2idx[layer_name]
        assert layer_idx is not None
        sp_idx, tm_idx = layer_idx

        # Allocate.
        if tm_idx == 0:
            start_list.append([])
        start_list[-1].append(0)
        assert sp_idx == len(start_list) - 1
        assert tm_idx == len(start_list[sp_idx]) - 1

        # Start time depends on the ready time of the previous on-chip layers.
        prev_layers, _ = self.network.prev_layers(layer_name)
        prev_indices = [self.layer2idx[pl] for pl in prev_layers
                        if pl in self.layer2idx]

        start = 0

        for prev_sp_idx, prev_tm_idx in prev_indices:
            if prev_sp_idx == sp_idx:
                # Same spatial scheduling, wait for full time.
                assert prev_tm_idx == tm_idx - 1, \
                        ('PipelineSegmentTiming: same-spatial dependency '
                         '{} of {} must be immediate previous.'
                         .format((prev_sp_idx, prev_tm_idx), layer_idx))
                prev_timing = self.timing_list[prev_sp_idx][prev_tm_idx]
                ready = start_list[prev_sp_idx][prev_tm_idx] \
                        + prev_timing.time // self.bat_ngrp
            else:
                assert prev_sp_idx < sp_idx
                # Previous spatial scheduling, wait for one group.
                assert prev_tm_idx == len(self.timing_list[prev_sp_idx]) - 1, \
                        ('PipelineSegmentTiming: previous-spatial dependency '
                         '{} must be the last temporal scheduling out of {}.'
                         .format((prev_sp_idx, prev_tm_idx),
                                 len(self.timing_list[prev_sp_idx])))
                # Number of groups should be min of input and previous output.
                ngrp = self.timing_list[sp_idx][tm_idx].ifm_ngrp
                tgrp = 0
                # Backtrace all fused layers.
                while True:
                    prev_timing = self.timing_list[prev_sp_idx][prev_tm_idx]
                    ngrp = min(ngrp, prev_timing.ofm_ngrp)
                    tgrp += prev_timing.time // self.bat_ngrp
                    if not prev_timing.fused or prev_tm_idx == 0:
                        ready = start_list[prev_sp_idx][prev_tm_idx] \
                                + tgrp // ngrp
                        break
                    prev_tm_idx -= 1

            # Max of dependency ready time.
            start = max(start, ready)

        start_list[sp_idx][tm_idx] = start

