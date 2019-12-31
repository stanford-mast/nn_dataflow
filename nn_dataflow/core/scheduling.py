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

from collections import OrderedDict, namedtuple
import math
import fastcache

from . import data_category_enum as de
from . import loop_blocking
from . import loop_enum as le
from . import mem_hier_enum as me
from . import partition
from .. import util
from .cost import Cost
from .data_layout import DataLayout
from .fmap_range import FmapPosition, FmapRange
from .layer import Layer
from .map_strategy import MapStrategy
from .resource import Resource
from .scheduling_constraint import SchedulingConstraint

class SchedulingCondition(namedtuple('SchedulingCondition',
                                     ['resource',
                                      'constraint',
                                      'ifmap_layout',
                                      'sched_seq',
                                     ])):
    '''
    Layer scheduling condition.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(SchedulingCondition, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.resource, Resource):
            raise TypeError('SchedulingCondition: resource must be '
                            'a Resource instance.')
        if not isinstance(ntp.constraint, SchedulingConstraint):
            raise TypeError('SchedulingCondition: constraint must be '
                            'a SchedulingConstraint instance.')
        if not isinstance(ntp.ifmap_layout, DataLayout):
            raise TypeError('SchedulingCondition: ifmap_layout must be '
                            'a DataLayout instance.')
        if not isinstance(ntp.sched_seq, tuple):
            raise TypeError('SchedulingCondition: sched_seq must be a tuple.')
        if len(ntp.sched_seq) != 3:
            raise ValueError('SchedulingCondition: sched_seq must have '
                             '(segment, spatial, temporal) 3 indices.')

        return ntp


class SchedulingResult(namedtuple('SchedulingResult',
                                  ['scheme',
                                   'ofmap_layout',
                                   'sched_seq',
                                  ])):
    '''
    Layer scheduling result.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(SchedulingResult, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.scheme, OrderedDict):
            raise TypeError('SchedulingResult: scheme must be an OrderedDict '
                            'instance.')
        if not isinstance(ntp.ofmap_layout, DataLayout):
            raise TypeError('SchedulingResult: ofmap_layout must be '
                            'a DataLayout instance.')
        if not isinstance(ntp.sched_seq, tuple):
            raise TypeError('SchedulingResult: sched_seq must be a tuple.')
        if len(ntp.sched_seq) != 3:
            raise ValueError('SchedulingResult: sched_seq must have '
                             '(segment, spatial, temporal) 3 indices.')

        return ntp

    @property
    def total_cost(self):
        ''' Get the total cost. '''
        return self.scheme['cost']

    @property
    def total_time(self):
        ''' Get the dataflow total time. '''
        return self.scheme['time']

    @property
    def total_node_time(self):
        ''' Get the total time on processing nodes. '''
        return max(self.scheme['proc_time'], self.scheme['bus_time'])

    @property
    def total_dram_time(self):
        ''' Get the total time on DRAM access. '''
        return self.scheme['dram_time']

    @property
    def total_proc_time(self):
        ''' Get the total active processing time. '''
        return self.scheme['proc_time']

    @property
    def total_ops(self):
        ''' Get the total ops. '''
        return self.scheme['ops']

    @property
    def total_accesses(self):
        ''' Get the total accesses at all memory hierarchies as a list. '''
        accesses = [sum(acc) for acc in self.scheme['access']]
        accesses[me.GBUF] += sum(self.scheme['remote_gbuf_access'])
        return accesses

    @property
    def total_noc_hops(self):
        ''' Get the total NoC hops. '''
        return sum(self.scheme['total_nhops'])

    @property
    def num_nodes(self):
        ''' Get the number of processing nodes. '''
        return self.scheme['num_nodes']


class Scheduling():
    '''
    Layer scheduling.
    '''

    def __init__(self, layer, batch_size, cost, map_strategy_class):

        if not isinstance(layer, Layer):
            raise TypeError('Scheduling: layer must be a Layer instance.')
        if not isinstance(cost, Cost):
            raise TypeError('Scheduling: cost must be a Cost instance.')

        if not issubclass(map_strategy_class, MapStrategy):
            raise TypeError('Scheduling: map_strategy_class must be '
                            'a subclass of MapStrategy.')

        self.layer = layer
        self.batch_size = batch_size
        self.cost = cost
        self.map_strategy_class = map_strategy_class

        # Default compare key function.
        self.cmp_key = lambda res: (res.total_cost, res.total_time)

    @fastcache.clru_cache(maxsize=1024)
    def schedule_search(self, condition, options):
        '''
        Search the best scheduling results.
        '''
        # Set key function.
        if options.opt_goal == 'ed':
            self.cmp_key = lambda res: res.total_cost * res.total_time
        elif options.opt_goal == 'd':
            self.cmp_key = lambda res: (res.total_time, res.total_cost)
        else:
            assert options.opt_goal == 'e'

        tops = []

        resource = condition.resource
        proc_region = resource.proc_region

        # Ifmap layout.
        ifmap_layout = condition.ifmap_layout
        # Ifmap should be from the source data region or local.
        if not ifmap_layout.is_in(resource.src_data_region, proc_region):
            raise ValueError('Scheduling: ifmap layout is not contained in '
                             'source data region.')
        ifrng = ifmap_layout.complete_fmap_range()
        if self.batch_size != ifrng.size('b') \
                or self.layer.nifm != ifrng.size('n') \
                or not self.layer.is_valid_padding_sifm([ifrng.size('h'),
                                                         ifrng.size('w')]):
            raise ValueError('Scheduling: ifmap layout does not match '
                             'input layer.')

        # Filter nodes. All memory nodes can store filters. Deduplicate.
        filter_nodes = frozenset(resource.dram_region.iter_node())

        # Explore parallel partitioning schemes.
        for part in partition.gen_partition(self.layer, self.batch_size,
                                            proc_region.dim, options,
                                            guaranteed=True):
            # Explore single-node schedules.
            lbs_tops = list(self.schedule_search_per_node(
                part, resource, condition.constraint, options))
            if not lbs_tops:
                continue

            # Ofmap layout.
            ofmap_range = FmapRange(
                FmapPosition(b=0, n=0, h=0, w=0),
                FmapPosition(b=self.batch_size, n=self.layer.nofm,
                             h=self.layer.hofm, w=self.layer.wofm))
            ofmap_data_region = resource.dst_data_region
            ofmap_layout = DataLayout(
                frngs=(ofmap_range,),
                regions=(ofmap_data_region,),
                parts=(part.projection(ofmap_data_region, appl2frng=True),))

            # Partition NoC hop cost.
            unit_nhops = partition.unit_nhops_to_proc_region(
                self.layer, self.batch_size, proc_region, part,
                filter_nodes, ifmap_layout, ofmap_layout, options)

            # Make scheduling result.
            tops += [self._get_result(lbs, part, ofmap_layout,
                                      condition.sched_seq, unit_nhops)
                     for lbs in lbs_tops]

        # Pick the top n.
        tops = sorted(tops, key=self.cmp_key)[:options.ntops]

        # Check total op count.
        total_layer_ops = self.layer.total_ops(self.batch_size)
        for t in tops:
            assert util.isclose(total_layer_ops, t.total_ops, rel_tol=1e-4)

        # Check ofmap layout matches the layer.
        for t in tops:
            ofrng = t.ofmap_layout.complete_fmap_range()
            assert ofrng.size('b') == self.batch_size \
                    and ofrng.size('n') == self.layer.nofm \
                    and ofrng.size('h') == self.layer.hofm \
                    and ofrng.size('w') == self.layer.wofm

        return list(tops)

    def cache_stats(self):
        '''
        Get the cache hits/misses stats. Return a tuple of (hits, misses).
        '''
        # pylint: disable=no-member
        info = self.schedule_search_per_node.cache_info()
        return (info.hits, info.misses)

    @fastcache.clru_cache(maxsize=1024)
    def schedule_search_per_node(self, part, resource, constraint, options):
        '''
        Search the best mapping strategies and loop blocking schemes for a
        single node after partitioning. Return the top LoopBlockingScheme
        instances.
        '''
        lbs_tops = []

        # Partitioned layer.
        p_layer, p_batch_size, p_occ = part.part_layer(self.layer,
                                                       self.batch_size)

        # Mapping strategy.
        map_strategy = self.map_strategy_class(p_layer, p_batch_size, p_occ,
                                               resource.dim_array)

        # Explore PE array mapping schemes for partitioned layer.
        for nested_loop_desc in map_strategy.gen_nested_loop_desc():

            # Explore loop blocking schemes.
            for lbs in loop_blocking.gen_loopblocking(
                    nested_loop_desc, resource, part, constraint, self.cost,
                    options):

                if lbs.is_valid():
                    lbs_tops.append(lbs)

        return lbs_tops

    def _get_result(self, lbs, part, ofmap_layout, sched_seq, unit_nhops):
        '''
        Make the schedule result from loop blocking and partitioning.
        '''
        scheme = OrderedDict()

        # Cost components.
        cost_access = lbs.get_access_cost(self.cost)

        # Inter-node data forwarding/rotation hops.
        node_nhops = lbs.get_noc_access()
        # Memory access hops.
        mem_nhops = [unh * f for unh, f
                     in zip(unit_nhops, lbs.get_top_level_fetch())]
        # Total hops = inter-node hops + memory hops.
        total_nhops = [nnh + mnh for nnh, mnh in zip(node_nhops, mem_nhops)]
        cost_noc = self.cost.noc_hop * sum(total_nhops)

        cost_op = self.cost.mac_op * lbs.ops

        cost_static = self.cost.idl_unit * lbs.time

        assert not math.isnan(cost_op + cost_access + cost_noc + cost_static)

        # Overall stats.
        scheme['cost'] = cost_op + cost_access + cost_noc + cost_static
        scheme['time'] = lbs.time
        scheme['ops'] = lbs.ops
        scheme['num_nodes'] = lbs.num_nodes
        scheme['is_dram'] = (lbs.src_is_dram, lbs.dst_is_dram)
        scheme['cost_op'] = cost_op
        scheme['cost_access'] = cost_access
        scheme['cost_noc'] = cost_noc
        scheme['cost_static'] = cost_static
        scheme['proc_time'] = lbs.proc_time
        scheme['bus_time'] = lbs.bus_time
        scheme['dram_time'] = lbs.dram_time
        scheme['access'] = lbs.get_access()
        scheme['remote_gbuf_access'] = lbs.remote_gbuf_access
        scheme['total_nhops'] = total_nhops
        scheme['fetch'] = lbs.fetch

        # Loop blocking.
        lp_ts = list(zip(*lbs.bl_ts))
        scheme['ti'] = tuple(lp_ts[le.IFM])
        scheme['to'] = tuple(lp_ts[le.OFM])
        scheme['tb'] = tuple(lp_ts[le.BAT])
        scheme['tvals'] = lbs.bl_ts
        scheme['orders'] = lbs.bl_ords
        scheme['size'] = [[lbs.data_size(bl, dce) for dce in range(de.NUM)]
                          for bl in range(lbs.BL.NUM)]
        scheme['unit_size'] = lbs.unit_size
        scheme['unit_cnt'] = lbs.unit_cnt
        scheme['accfwd_reduction'] = lbs.accfwd_reduction
        scheme['bufshr_grp_size'] = lbs.bufshr_grp_size
        scheme['bufshr_subgrp_size'] = lbs.bufshr_subgrp_size
        scheme['bufshr_bs_t'] = lbs.bufshr_bs_t
        scheme['bufshr_bs_ord'] = lbs.bufshr_bs_ord
        scheme['bufshr_rot_fetch'] = lbs.bufshr_rot_fetch
        scheme['bufshr_rot_round_cnt'] = lbs.bufshr_rot_round_cnt
        scheme['bufshr_rot_unit_cnt'] = lbs.bufshr_rot_unit_cnt
        scheme['bufshr_wide_fetch'] = lbs.bufshr_wide_fetch
        scheme['bufshr_wide_fetch_width'] = lbs.bufshr_wide_fetch_width

        # Partitioning.
        scheme['part'] = part
        scheme['mem_nhops'] = mem_nhops
        scheme['node_nhops'] = node_nhops
        scheme['unit_nhops'] = unit_nhops

        return SchedulingResult(scheme=scheme, ofmap_layout=ofmap_layout,
                                sched_seq=sched_seq)

