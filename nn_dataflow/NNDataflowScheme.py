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
from collections import OrderedDict, MutableMapping

from . import LoopEnum as le
from . import MemHierEnum as me
from . import Util
from .DataLayout import DataLayout
from .Layer import ConvLayer
from .Network import Network
from .Scheduling import SchedulingResult

class SegmentScheduleTiming(object):
    ''' Timing information of a segment schedule. '''

    class SpatialScheduleTiming(object):
        ''' Time information of a spatial schedule. '''

        def __init__(self, head=0, tail=0, tail_split=float('inf')):
            self.head = head
            self.tail = tail
            self.tail_split = tail_split

        def reset_tail(self, tail, tail_split):
            ''' Add current tail to head and reset tail. '''
            self.head += self.tail
            self.tail = tail
            self.tail_split = tail_split

        def incr_tail(self, tail):
            ''' Increment tail. '''
            self.tail += tail

        def time(self):
            ''' Total time. '''
            return self.head + self.tail

        def split_time(self, tbat_split):
            ''' Time with split tail. '''
            return self.head + self.tail / self.tail_split / tbat_split

        def __repr__(self):
            return '{}(head={}, tail={}, tail_split={})' \
                    .format(self.__class__.__name__, self.head, self.tail,
                            self.tail_split)


    def __init__(self, seg_idx, sp_timing_list=None, tbat_split=float('inf'),
                 last_sched_seq=None):

        # Segment index.
        self.seg_idx = seg_idx

        # A list of spatial schedule timing info in this segment.
        self.sp_timing_list = sp_timing_list if sp_timing_list else []

        # The top BAT loop factor, shared by all layers in the segment.
        self.tbat_split = tbat_split

        # Schedule sequence number of the last added schedule.
        self.last_sched_seq = last_sched_seq if last_sched_seq else tuple()

    def time(self):
        ''' The total time of the segment schedule. '''

        # The longest spatial schedule in the segment.
        max_idx, max_sp_timing = max(enumerate(self.sp_timing_list),
                                     key=lambda tpl: tpl[1].time())

        t = 0

        # Segment pipeline filling/draining time.
        t += sum(sp_timing.split_time(self.tbat_split)
                 for sp_timing in self.sp_timing_list[:-1])
        # Segment critical stage time (minus filling/draining time).
        t += max_sp_timing.time()

        if max_idx != len(self.sp_timing_list) - 1:
            # Minus the double counted part.
            t -= max_sp_timing.split_time(self.tbat_split)

        return t

    def add(self, layer, sched_result):
        ''' Add the SchedulingResult of a new layer. '''

        # Schedule sequence number.
        sched_seq = sched_result.sched_seq

        # Loop blocking scheme: blocking factors and orders.
        lp_ts = [None] * le.NUM
        lp_ts[le.IFM] = sched_result.dict_loop['ti']
        lp_ts[le.OFM] = sched_result.dict_loop['to']
        lp_ts[le.BAT] = sched_result.dict_loop['tb']
        bl_ts = list(zip(*lp_ts))
        bl_ords = sched_result.dict_loop['orders']

        # Check sched_seq, and find the incremented dimension.
        if sched_seq[0] != self.seg_idx:
            raise ValueError('SegmentScheduleTiming: sched_seq {} does not '
                             'belong to segment {}'
                             .format(sched_seq, self.seg_idx))

        if not self.last_sched_seq:
            if sched_seq[1] != 0 or sched_seq[2] != 0:
                raise ValueError('SegmentScheduleTiming: sched_seq {} cannot '
                                 'be the first schedule in segment {}'
                                 .format(sched_seq, self.seg_idx))
            incr_pos = 1

        else:
            incr_pos = 1 if sched_seq[1] != self.last_sched_seq[1] else 2

            if sched_seq[incr_pos] != self.last_sched_seq[incr_pos] + 1:
                raise ValueError('SegmentScheduleTiming: sched_seq {} cannot '
                                 'follow {}'
                                 .format(sched_seq, self.last_sched_seq))

        assert 1 <= incr_pos <= 2

        if incr_pos == 1:
            self.sp_timing_list.append(
                SegmentScheduleTiming.SpatialScheduleTiming())
        assert sched_seq[1] + 1 == len(self.sp_timing_list)

        self.last_sched_seq = sched_seq

        sp_timing = self.sp_timing_list[-1]

        if isinstance(layer, ConvLayer):
            # Conv layer.

            # Ordered loops from outer to inner with LoopEnum values and
            # blocking factors.
            ord_loops = []
            for bl_t, bl_ord in zip(bl_ts, bl_ords):
                lps = [None] * le.NUM
                for lpe in range(le.NUM):
                    lps[le.NUM - 1 - bl_ord[lpe]] = (lpe, bl_t[lpe])
                assert None not in lps
                ord_loops += lps
            # Skip trivial loops.
            ord_loops = [(lpe, t) for lpe, t in ord_loops if t > 1]

            # Top BAT split factor.
            tbat_split = 1.
            for lpe, t in ord_loops:
                if lpe == le.BAT:
                    tbat_split *= t
                else:
                    break

            # Tail split factor.
            tail_split = 1.
            for lpe, t in itertools.dropwhile(lambda tpl: tpl[0] == le.BAT,
                                              ord_loops):
                if lpe == le.OFM:
                    tail_split *= t
                else:
                    break

            self.tbat_split = min(self.tbat_split, tbat_split)
            sp_timing.reset_tail(sched_result.total_time, tail_split)

        else:
            # Not Conv layer, accumulate tail time.
            sp_timing.incr_tail(sched_result.total_time)

    def __repr__(self):
        return '{}(seg_idx={}, sp_timing_list={}, tbat_split={}, ' \
               'last_sched_seq={})'.format(self.__class__.__name__,
                                           self.seg_idx,
                                           self.sp_timing_list,
                                           self.tbat_split,
                                           self.last_sched_seq)


class NNDataflowScheme(MutableMapping):
    '''
    Neural network dataflow result, as a specialized OrderedDict of layer
    scheduling results.
    '''

    def __init__(self, network, input_layout):
        # pylint: disable=super-init-not-called

        if not isinstance(network, Network):
            raise TypeError('NNDataflowScheme: network must be a '
                            'Network instance.')
        if not isinstance(input_layout, DataLayout):
            raise TypeError('NNDataflowScheme: input_layout must be a '
                            'DataLayout instance.')
        self.network = network
        self.input_layout = input_layout

        self.res_dict = OrderedDict()

        self.total_cost = 0

        # A list of segment schedule timing information.
        self.seg_timing_list = []
        # A list of segment time.
        self.seg_time_list = []

        self.last_sched_seq = (-1, 0, 0)

    def __getitem__(self, layer_name):
        ''' Get the SchedulingResult of a scheduled layer. '''
        return self.res_dict[layer_name]

    def __setitem__(self, layer_name, sched_result):
        ''' Add the SchedulingResult of a new layer. '''
        if layer_name not in self.network:
            raise KeyError('NNDataflowScheme: layer {} does not belong to '
                           'network {}.'
                           .format(layer_name, self.network.net_name))
        if layer_name in self.res_dict:
            raise KeyError('NNDataflowScheme: layer {} already exists.'
                           .format(layer_name))
        if not isinstance(sched_result, SchedulingResult):
            raise TypeError('NNDataflowScheme: sched_result must be '
                            'a SchedulingResult instance.')

        prev_layers, _ = self.network.prev_layers(layer_name)
        for pl in prev_layers:
            if pl is None:
                continue
            if pl not in self.res_dict:
                raise KeyError('NNDataflowScheme: layer {} has its previous '
                               'layer {} not scheduled yet.'
                               .format(layer_name, pl))

        self.res_dict[layer_name] = sched_result
        self.total_cost += sched_result.total_cost
        self._update_time(layer_name, sched_result)
        self.last_sched_seq = sched_result.sched_seq

    def __delitem__(self, layer_name):
        ''' Not legal to call. '''
        raise KeyError('NNDataflowScheme: cannot delete scheduled layer.')

    def __iter__(self):
        ''' Iterate over scheduled layers. '''
        for layer_name in self.res_dict:
            yield layer_name

    def __len__(self):
        ''' Get the number of scheduled layers. '''
        return len(self.res_dict)

    def copy(self):
        '''
        Return a shallow copy.

        Shallow copy of layer SchedulingResult is sufficient, since they are
        read-only.
        '''
        df = NNDataflowScheme(self.network, self.input_layout)
        for layer_name in self.res_dict:
            df[layer_name] = self.res_dict[layer_name]
        assert Util.isclose(df.total_cost, self.total_cost, rel_tol=1e-5)
        assert Util.isclose(df.total_time, self.total_time, rel_tol=1e-5)
        return df

    @property
    def total_time(self):
        ''' Get the total time. '''
        return sum(self.seg_time_list)

    @property
    def total_ops(self):
        ''' Get the total ops. '''
        return sum(sr.total_ops for sr in self.values())

    @property
    def total_accesses(self):
        ''' Get the total accesses at all memory hierarchies as a list. '''
        accesses = [0] * me.NUM
        for sr in self.values():
            accesses = [a + a1 for a, a1 in zip(accesses, sr.total_accesses)]
        return accesses

    @property
    def total_noc_hops(self):
        ''' Get the total NoC hops. '''
        return sum(sr.total_noc_hops for sr in self.values())

    @property
    def total_node_time(self):
        ''' Get the total node-time products. '''
        return sum(sr.total_time * sr.dict_part['num_nodes']
                   for sr in self.values())

    def perlayer_stats(self, stats_name):
        '''
        Get a dict of per-layer stats. Valid stats must be a static method.
        '''
        try:
            stats_func = getattr(self, stats_name)
        except AttributeError:
            raise AttributeError('NNDataflowScheme: stats {} is not supported.'
                                 .format(stats_name))

        stats = OrderedDict()
        for layer_name in self.res_dict:
            stats[layer_name] = stats_func(self.res_dict[layer_name])

        return stats

    @staticmethod
    def active_node_pes(sched_result):
        ''' Layer active node PE counts. '''
        return 1. * sched_result.total_ops / sched_result.total_time \
                / sched_result.dict_part['num_nodes']

    @staticmethod
    def total_dram_bandwidth(sched_result):
        ''' Layer total DRAM bandwidth in elements per cycle. '''
        return 1. * sched_result.total_accesses[me.DRAM] \
                / sched_result.total_time

    def _update_time(self, layer_name, sched_result):
        '''
        Update the total time.
        '''

        seg_idx = sched_result.sched_seq[0]

        try:
            timing = self.seg_timing_list[seg_idx]
        except IndexError:
            # A new segment.
            if seg_idx != len(self.seg_timing_list):
                raise ValueError('NNDataflowScheme: segment {} cannot follow '
                                 'segment {}'
                                 .format(seg_idx, len(self.seg_timing_list)))
            timing = SegmentScheduleTiming(seg_idx)
            self.seg_timing_list.append(timing)
            self.seg_time_list.append(0)

        timing.add(self.network[layer_name], sched_result)

        self.seg_time_list[-1] = timing.time()

