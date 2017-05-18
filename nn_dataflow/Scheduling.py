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

import math
from collections import OrderedDict, namedtuple

from . import LoopBlocking
from . import Partition
from .Cost import Cost
from .DataLayout import DataLayout
from .Layer import Layer
from .MapStrategy import MapStrategy
from .Resource import Resource

class SchedulingCondition(namedtuple('SchedulingCondition',
                                     ['resource',
                                      'ifmap_layout',
                                     ])):
    '''
    Layer scheduling condition (constraints).
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(SchedulingCondition, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.resource, Resource):
            raise TypeError('SchedulingCondition: resource must be '
                            'a Resource instance.')
        if not isinstance(ntp.ifmap_layout, DataLayout):
            raise TypeError('SchedulingCondition: ifmap_layout must be '
                            'a DataLayout instance.')

        return ntp


class SchedulingResult(namedtuple('SchedulingResult',
                                  ['total_cost',
                                   'dict_loop',
                                   'dict_part',
                                   'ofmap_layout',
                                  ])):
    '''
    Layer scheduling result.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(SchedulingResult, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.dict_loop, OrderedDict) \
                or not isinstance(ntp.dict_part, OrderedDict):
            raise TypeError('SchedulingResult: dict_loop and dict_part '
                            'must be OrderedDict instances.')
        if not isinstance(ntp.ofmap_layout, DataLayout):
            raise TypeError('SchedulingCondition: ofmap_layout must be '
                            'a DataLayout instance.')

        return ntp


class Scheduling(object):
    '''
    Layer scheduling.
    '''
    # pylint: disable=too-many-instance-attributes

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

        # Per-node schedule cache.
        self.pernode_sched_cache = {}
        self.pernode_sched_cache_hits = 0
        self.pernode_sched_cache_misses = 0

    def schedule_search(self, condition, options):
        '''
        Search the best schedule results under the given condition and options.
        '''
        tops = []

        mem_region_src = condition.resource.mem_region_src()
        mem_region_dst = condition.resource.mem_region_dst()

        # Ifmap layout.
        ifmap_layout = condition.ifmap_layout
        if not ifmap_layout.is_in_region(mem_region_src):
            raise ValueError('Scheduling: ifmap layout contains invalid '
                             'source memory nodes.')

        # Filter nodes. All memory nodes can store filters. Deduplicate.
        filter_node_coord_list = [c for c in mem_region_src.node_iter()] \
                               + [c for c in mem_region_dst.node_iter()]
        filter_node_coord_list = list(set(filter_node_coord_list))

        # Explore parallel partitioning schemes.
        for part in Partition.gen_partition(self.layer, self.batch_size,
                                            condition.resource.dim_nodes,
                                            options):
            # Ofmap layout.
            ofmap_layout = Partition.get_ofmap_layout(
                self.layer, self.batch_size, part, mem_region_dst)

            # Partition NoC hop cost.
            unit_nhops = Partition.part_layer_unit_nhops(
                self.layer, self.batch_size, part, filter_node_coord_list,
                ifmap_layout, ofmap_layout, options)
            if math.isinf(sum(unit_nhops)):
                continue

            # Explore single-node schedules.
            for lbs in self.schedule_search_per_node(
                    part, condition.resource, options):

                # Make scheduling result.
                r = self._get_result(lbs, part, unit_nhops, ofmap_layout,
                                     condition, options)
                tops.append(r)

        # Pick the top n.
        tops = sorted(tops, key=lambda r: r.total_cost)[:options.ntops]

        # Check total op count.
        # Initial occupation also applies to layer.
        total_layer_ops = self.layer.total_ops(self.batch_size)
        for t in tops:
            sum_part_layer_ops = t.dict_loop['ops'] \
                    * condition.resource.dim_nodes.size()
            assert abs(float(total_layer_ops) / sum_part_layer_ops - 1) < 1e-4

        # Check ofmap layout matches the layer.
        for t in tops:
            cofrng = t.ofmap_layout.frmap.complete_fmap_range()
            b_rng, n_rng, h_rng, w_rng = cofrng.beg_end()
            assert b_rng[1] - b_rng[0] == self.batch_size \
                    and n_rng[1] - n_rng[0] == self.layer.nofm \
                    and h_rng[1] - h_rng[0] == self.layer.hofm \
                    and w_rng[1] - w_rng[0] == self.layer.wofm

        return list(tops)

    def cache_stats(self):
        '''
        Get the cache hits/misses stats. Return a tuple of (hits, misses).
        '''
        return (self.pernode_sched_cache_hits, self.pernode_sched_cache_misses)

    def schedule_search_per_node(self, part, resource, options):
        '''
        Search the best mapping strategies and loop blocking schemes for a
        single node after partitioning, given the partitioning scheme and
        resource.

        Return the top LoopBlockingScheme instances.
        '''

        # NOTE: need to ensure the key's __eq__ and __hash__ have been
        # redefined.
        cache_key = (part, resource, options)

        cache_val = self.pernode_sched_cache.get(cache_key, None)

        if cache_val is not None:
            # Cache hit.
            self.pernode_sched_cache_hits += 1
            return cache_val

        # Cache miss.
        self.pernode_sched_cache_misses += 1

        top_lbs_list = []

        # Partitioned layer.
        p_layer, p_batch_size, p_occ = part.part_layer(self.layer,
                                                       self.batch_size)

        # Mapping strategy.
        map_strategy = self.map_strategy_class(p_layer, p_batch_size,
                                               resource.dim_array)

        # Explore PE array mapping schemes for partitioned layer.
        for nested_loop_desc in map_strategy.gen_nested_loop_desc():

            # Explore loop blocking schemes.
            for lbs in LoopBlocking.gen_loopblocking(
                    nested_loop_desc, resource, self.cost, p_occ, options):

                if lbs.is_valid():
                    top_lbs_list.append(lbs)

        self.pernode_sched_cache[cache_key] = top_lbs_list

        return top_lbs_list

    def _get_result(self, lbs, part, unit_nhops, ofmap_layout, condition,
                    options):
        '''
        Make the schedule result from loop blocking and partitioning.
        '''
        del options  # unused

        # Loop blocking.
        cost_loop = lbs.get_cost(self.cost)
        dict_loop = OrderedDict([('cost', cost_loop)])
        dict_loop.update(lbs.get_scheme_dict())

        # Partitioning.
        total_nhops = [unh * f
                       for unh, f
                       in zip(unit_nhops,
                              lbs.get_top_level_fetch())]
        cost_part = self.cost.noc_hop * sum(total_nhops)
        dict_part = OrderedDict([('cost', cost_part),
                                 ('total_nhops', total_nhops),
                                 ('part', part.__dict__),
                                 ('unit_nhops', unit_nhops)])

        # Result.
        total_cost = cost_loop * condition.resource.dim_nodes.size() + cost_part

        return SchedulingResult(total_cost=total_cost,
                                dict_loop=dict_loop,
                                dict_part=dict_part,
                                ofmap_layout=ofmap_layout)

