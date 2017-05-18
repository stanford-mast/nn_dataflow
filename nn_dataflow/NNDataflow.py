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

import sys
from collections import OrderedDict

from . import Partition
from .Cost import Cost
from .Network import Network
from .Resource import Resource
from .Scheduling import SchedulingCondition, SchedulingResult, Scheduling

class SchedulingResultDict(object):
    '''
    Network scheduling result, as a dict of layer scheduling results.

    Include the total cost, and the layer scheduling results as an OrderedDict.
    '''

    def __init__(self, res_dict=None):

        total_cost = 0

        if res_dict is None:
            res_dict = OrderedDict()
        else:
            for name in res_dict:
                res = res_dict[name]
                if isinstance(res, SchedulingResult):
                    total_cost += res.total_cost
                else:
                    raise TypeError('SchedulingResultDict: res_dict value type '
                                    'must be SchedulingResult.')

        self.total_cost = total_cost
        self.res_dict = res_dict

    def __len__(self):
        ''' Get the number of scheduled layers. '''
        return len(self.res_dict)

    def __getitem__(self, layer_name):
        ''' Get the layer SchedulingResult. '''
        return self.res_dict[layer_name]

    def __setitem__(self, layer_name, sched_result):
        ''' In-place update by adding the result of a new layer. '''
        if layer_name in self.res_dict:
            raise KeyError('SchedulingResultDict: layer {} already exists.'
                           .format(layer_name))
        if not isinstance(sched_result, SchedulingResult):
            raise TypeError('SchedulingResultDict: sched_result must be '
                            'a SchedulingResult instance.')
        self.total_cost += sched_result.total_cost
        self.res_dict[layer_name] = sched_result

    def __contains__(self, layer_name):
        ''' Whether the layer is already scheduled. '''
        return layer_name in self.res_dict

    def scheduling_total_cost(self):
        ''' Get the scheduling total cost. '''
        return self.total_cost

    def scheduling_result_dict(self):
        ''' Get the scheduling result dict. '''
        return self.res_dict

    def copy(self):
        ''' Return a shallow copy. '''
        # Shallow copy of layer SchedulingResult is sufficient, since they are
        # read-only.
        return SchedulingResultDict(self.res_dict.copy())

    def __cmp__(self, other):
        if not isinstance(other, SchedulingResultDict):
            raise TypeError('SchedulingResultDict: a SchedulingResultDict '
                            'object is required.')
        if self.total_cost > other.total_cost:
            return 1
        elif self.total_cost < other.total_cost:
            return -1
        return 0


class NNDataflow(object):
    '''
    Search optimized dataflows for neural networks.
    '''
    # pylint: disable=too-few-public-methods

    def __init__(self, network, batch_size, resource, cost):
        if not isinstance(network, Network):
            raise TypeError('NNDataflow: network must be a Network instance.')
        if not isinstance(resource, Resource):
            raise TypeError('NNDataflow: resource must be a Resource instance.')
        if not isinstance(cost, Cost):
            raise TypeError('NNDataflow: cost must be a Cost instance.')

        self.network = network
        self.batch_size = batch_size
        self.resource = resource
        self.cost = cost

        # Dict of layer Scheduling instances.
        self.layer_sched_dict = {}

    def schedule_search(self, map_strategy_class, options):
        '''
        Search the optimized dataflows.
        '''

        # Scheduling instance dict.
        self.layer_sched_dict = {}
        layer2sched = {}
        for layer_name in self.network:
            layer = self.network[layer_name]
            sched = layer2sched.get(layer, None)
            if sched is None:
                sched = Scheduling(layer, self.batch_size, self.cost,
                                   map_strategy_class)
                layer2sched[layer] = sched
            self.layer_sched_dict[layer_name] = sched

        # Initial input layout.
        sched_res_dict_list = []
        for input_layout in self._gen_input_layout(options):
            srd = SchedulingResultDict()
            srd[self.network.INPUT_LAYER_KEY] = SchedulingResult(
                total_cost=0, dict_loop=OrderedDict(), dict_part=OrderedDict(),
                ofmap_layout=input_layout)
            sched_res_dict_list.append(srd)

        for layer_name in self.network:
            if options.verbose:
                sys.stderr.write('-> {}\n'.format(layer_name))
                sys.stderr.flush()
            sched_res_dict_list = self._layer_schedule_search(
                layer_name, sched_res_dict_list, options)

        # Cache stats.
        cache_hits = 0
        cache_misses = 0
        for sched in layer2sched.values():
            h, m = sched.cache_stats()
            cache_hits += h
            cache_misses += m

        return sched_res_dict_list, (cache_hits, cache_misses)

    def _layer_schedule_search(self, layer_name, sched_res_dict_list, options):
        '''
        Schedule the given layer under the previous layer scheduling results.
        `sched_res_dict_list` contains up to top n SchedulingResultDict for the
        previous layers.
        '''

        layer_sched = self.layer_sched_dict[layer_name]

        new_sched_res_dict_list = []

        for ifmap_layout, srd_idx in self._gen_layer_ifmap_layout(
                layer_name, sched_res_dict_list, options):

            condition = SchedulingCondition(resource=self.resource,
                                            ifmap_layout=ifmap_layout)

            try:
                tops = layer_sched.schedule_search(condition, options)
            except Exception:
                sys.stderr.write('Failed when scheduling layer {}.\n'
                                 .format(layer_name))
                raise

            if not tops:
                sys.stderr.write('Layer {} has no valid schedule.\n'
                                 .format(layer_name))

            # Append all the current layer top schedules to all the previous top
            # schedules with the matching fmap layout.
            for t in tops:
                srd = sched_res_dict_list[srd_idx].copy()
                srd[layer_name] = t
                new_sched_res_dict_list.append(srd)

        # Always pick and keep top n at each layer.
        return sorted(new_sched_res_dict_list)[:options.ntops]

    def _gen_layer_ifmap_layout(self, layer_name, sched_res_dict_list, options):
        '''
        Generator to get all the choices of ifmap layout for the layer.

        Return the ifmap layout, and the corresponding SchedulingResultDict
        index in the list.
        '''

        del options

        prev_layer_names, merge_symbol = self.network.prev_layers(layer_name)
        assert prev_layer_names

        for idx, srd in enumerate(sched_res_dict_list):
            # Merge all previous layer ofmap layouts to get the ifmap layout.
            ifmap_layout = srd[prev_layer_names[0]].ofmap_layout
            for pl_name in prev_layer_names[1:]:
                ifmap_layout = ifmap_layout.merge(merge_symbol,
                                                  srd[pl_name].ofmap_layout)

            # Remap dst memory to src memory.
            origin_diff = self.resource.mem_region_src().origin \
                    - self.resource.mem_region_dst().origin
            ifmap_layout = ifmap_layout.view(origin_diff=origin_diff)

            # We already checked the ofmap layout dimension in Scheduling, and
            # the prev/next layer dimensions in Network, so ifmap_layout ==
            # layer == prev_layers == ofmap_layout.

            yield ifmap_layout, idx

    def _gen_input_layout(self, options):
        '''
        Get the input layer layout choices.
        '''

        input_layer = self.network.input_layer()

        mem_region = self.resource.mem_region_dst()

        for part in Partition.gen_partition(input_layer, self.batch_size,
                                            mem_region.dim, options):
            input_layout = Partition.get_ofmap_layout(
                input_layer, self.batch_size, part, mem_region)

            yield input_layout

