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
from .Layer import Layer
from .Network import Network
from .PartitionScheme import PartitionScheme
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

    def schedule_search(self, map_strategy_class, options):
        '''
        Search the optimized dataflows.
        '''

        aggr_tops = [SchedulingResultDict()]
        part_lprev_list = list(self._gen_input_part(options))
        aggr_top_indexes_list = [[0] for _ in range(len(part_lprev_list))]

        for layer_name in self.network:
            aggr_tops = self._layer_schedule_search(
                layer_name, aggr_tops, part_lprev_list, aggr_top_indexes_list,
                map_strategy_class, options)

        return aggr_tops

    def _layer_schedule_search(self, layer_name, aggr_tops, part_lprev_list,
                               aggr_top_indexes_list, map_strategy_class,
                               options):
        '''
        Schedule the given layer under the previous layer scheduling results.
        '''

        layer = self.network[layer_name]
        layer_sched = Scheduling(layer, self.batch_size, self.cost,
                                 map_strategy_class)

        new_aggr_tops = []

        # For each previous layer partition scheme, search top schedules for
        # the current layer.
        for part_lprev, aggr_top_indexes in zip(part_lprev_list,
                                                aggr_top_indexes_list):

            condition = SchedulingCondition(resource=self.resource,
                                            part_src=part_lprev)

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
            # schedules with the matching partition scheme.
            for t_idx in range(options.ntops):
                if t_idx >= len(tops):
                    break
                assert tops[t_idx].dict_part['part_src'] == part_lprev.__dict__
                for at_idx in aggr_top_indexes:
                    atop = aggr_tops[at_idx].copy()
                    atop[layer_name] = tops[t_idx]
                    new_aggr_tops.append(atop)

        # Always pick and keep top n at each layer.
        aggr_tops = sorted(new_aggr_tops)[:options.ntops]

        # Record all layer partition schemes for next layer.
        part_lprev_list = []
        aggr_top_indexes_list = []
        for at_idx in range(options.ntops):
            if at_idx >= len(aggr_tops):
                break
            # Translate back to PartitionScheme.
            part_lprev_dict = aggr_tops[at_idx][layer_name].dict_part['part_dst']
            part_lprev = PartitionScheme(part_lprev_dict['order'],
                                         part_lprev_dict['pdims'])
            try:
                i = part_lprev_list.index(part_lprev)
            except ValueError:
                assert part_lprev_list.count(part_lprev) == 0
                part_lprev_list.append(part_lprev)
                aggr_top_indexes_list.append([])
                assert len(part_lprev_list) == len(aggr_top_indexes_list)
                i = -1
            aggr_top_indexes_list[i].append(at_idx)

        return aggr_tops

    def _gen_input_part(self, options):
        '''
        Get the input layer partitioning schemes.
        '''

        first_layer = self.network[self.network.first_layer_name()]
        input_layer = Layer(nifm=1, nofm=first_layer.nifm,
                            sofm=(first_layer.hifm, first_layer.wifm), sfil=1)

        mem_region = self.resource.mem_region_src()

        for part in Partition.gen_partition(input_layer, self.batch_size,
                                            mem_region.dim, options):
            yield part

