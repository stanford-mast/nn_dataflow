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
from collections import OrderedDict, namedtuple

from . import loop_blocking
from . import partition
from .. import util
from .cost import Cost
from .data_layout import DataLayout
from .layer import Layer
from .map_strategy import MapStrategy
from .resource import Resource

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
                                  ['dict_loop',
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
            raise TypeError('SchedulingResult: ofmap_layout must be '
                            'a DataLayout instance.')

        return ntp

    @property
    def total_cost(self):
        ''' Get the total cost. '''
        return self.dict_loop['cost'] + self.dict_part['cost']

    @property
    def total_time(self):
        ''' Get the dataflow total time. '''
        return self.dict_loop['time']

    @property
    def total_ops(self):
        ''' Get the total ops. '''
        # dict_loop stats are over all nodes.
        return self.dict_loop['ops']

    @property
    def total_accesses(self):
        ''' Get the total accesses at all memory hierarchies as a list. '''
        # dict_loop stats are over all nodes.
        return [sum(acc) for acc in self.dict_loop['access']]

    @property
    def total_noc_hops(self):
        ''' Get the total NoC hops. '''
        return sum(self.dict_part['total_nhops'])


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

        proc_region = condition.resource.proc_region
        src_data_region = condition.resource.src_data_region()
        dst_data_region = condition.resource.dst_data_region()

        # Ifmap layout.
        ifmap_layout = condition.ifmap_layout
        if not ifmap_layout.is_in_region(src_data_region):
            raise ValueError('Scheduling: ifmap layout contains invalid '
                             'source memory nodes.')
        cifrng = ifmap_layout.frmap.complete_fmap_range()
        if cifrng.size('b') != self.batch_size \
                or cifrng.size('n') != self.layer.nifm \
                or not self.layer.is_valid_padding_sifm([cifrng.size('h'),
                                                         cifrng.size('w')]):
            raise ValueError('Scheduling: ifmap layout does not match '
                             'input layer.')

        # Filter nodes. All memory nodes can store filters. Deduplicate.
        filter_nodes = set(itertools.chain(src_data_region.node_iter(),
                                           dst_data_region.node_iter()))

        # Explore parallel partitioning schemes.
        for part in partition.gen_partition(self.layer, self.batch_size,
                                            proc_region.dim, options,
                                            guaranteed=True):
            # Ofmap layout.
            ofmap_layout = partition.get_ofmap_layout(
                self.layer, self.batch_size, part, dst_data_region)

            # Partition NoC hop cost.
            unit_nhops = partition.part_layer_unit_nhops(
                self.layer, self.batch_size, part, proc_region,
                filter_nodes, ifmap_layout, ofmap_layout, options)

            # Explore single-node schedules.
            for lbs in self.schedule_search_per_node(
                    part, condition.resource, options):

                # Make scheduling result.
                r = self._get_result(lbs, part, unit_nhops, ofmap_layout)
                tops.append(r)

        # Pick the top n.
        tops = sorted(tops, key=lambda r: r.total_cost)[:options.ntops]

        # Check total op count.
        # Initial occupation also applies to layer.
        total_layer_ops = self.layer.total_ops(self.batch_size)
        for t in tops:
            actual_layer_ops = t.dict_loop['ops']
            assert util.isclose(total_layer_ops, actual_layer_ops, rel_tol=1e-4)

        # Check ofmap layout matches the layer.
        for t in tops:
            cofrng = t.ofmap_layout.frmap.complete_fmap_range()
            assert cofrng.size('b') == self.batch_size \
                    and cofrng.size('n') == self.layer.nofm \
                    and cofrng.size('h') == self.layer.hofm \
                    and cofrng.size('w') == self.layer.wofm

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
            for lbs in loop_blocking.gen_loopblocking(
                    nested_loop_desc, resource, self.cost, p_occ, options):

                if lbs.is_valid():
                    top_lbs_list.append(lbs)

        self.pernode_sched_cache[cache_key] = top_lbs_list

        return top_lbs_list

    def _get_result(self, lbs, part, unit_nhops, ofmap_layout):
        '''
        Make the schedule result from loop blocking and partitioning.
        '''

        # Loop blocking.
        dict_loop = lbs.get_scheme_dict(self.cost)

        # Partitioning.
        total_nhops = [unh * f for unh, f
                       in zip(unit_nhops, lbs.get_top_level_fetch())]
        cost_part = self.cost.noc_hop * sum(total_nhops)
        dict_part = OrderedDict([('cost', cost_part),
                                 ('num_nodes', part.size()),
                                 ('total_nhops', total_nhops),
                                 ('part', part.__dict__),
                                 ('unit_nhops', unit_nhops)])

        return SchedulingResult(dict_loop=dict_loop,
                                dict_part=dict_part,
                                ofmap_layout=ofmap_layout)

