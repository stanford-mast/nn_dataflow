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

from . import partition
from .cost import Cost
from .map_strategy import MapStrategy
from .network import Network
from .nn_dataflow_scheme import NNDataflowScheme
from .resource import Resource
from .scheduling import SchedulingCondition, Scheduling

class NNDataflow(object):
    '''
    Search optimized dataflows for neural networks.
    '''
    # pylint: disable=too-few-public-methods

    def __init__(self, network, batch_size, resource, cost, map_strategy):
        if not isinstance(network, Network):
            raise TypeError('NNDataflow: network must be a Network instance.')
        if not isinstance(resource, Resource):
            raise TypeError('NNDataflow: resource must be a Resource instance.')
        if not isinstance(cost, Cost):
            raise TypeError('NNDataflow: cost must be a Cost instance.')
        if not issubclass(map_strategy, MapStrategy):
            raise TypeError('NNDataflow: map_strategy must be a subclass of '
                            'MapStrategy.')

        self.network = network
        self.batch_size = batch_size
        self.resource = resource
        self.cost = cost
        self.map_strategy = map_strategy

        # Dict of layer Scheduling instances.
        self.layer_sched_dict = {}

    def schedule_search(self, options):
        '''
        Search the optimized dataflows.
        '''
        # Scheduling instance dict. Use the same instance for all same layers
        # in order to exploit its scheduling cache.
        self.layer_sched_dict = {}
        layer2sched = {}
        for layer_name in self.network:
            layer = self.network[layer_name]
            sched = layer2sched.get(layer, None)
            if sched is None:
                sched = Scheduling(layer, self.batch_size, self.cost,
                                   self.map_strategy)
                layer2sched[layer] = sched
            self.layer_sched_dict[layer_name] = sched

        # Initial input layout.
        dfsch_list = []
        for input_layout in self._gen_input_layout(options):
            dfsch = NNDataflowScheme(self.network, input_layout)
            dfsch_list.append(dfsch)

        # Schedule layers.
        for layer_name in self.network:
            if options.verbose:
                sys.stderr.write('-> {}\n'.format(layer_name))
                sys.stderr.flush()
            dfsch_list = self._layer_schedule_search(
                layer_name, dfsch_list, options)

        # Cache stats.
        cache_hits = 0
        cache_misses = 0
        for sched in layer2sched.values():
            h, m = sched.cache_stats()
            cache_hits += h
            cache_misses += m

        return dfsch_list, (cache_hits, cache_misses)

    def _layer_schedule_search(self, layer_name, dfsch_list, options):
        '''
        Schedule the given layer under the previous layer scheduling results.
        `dfsch_list` contains up to top n NNDataflowScheme for the previous
        layers.
        '''

        layer_sched = self.layer_sched_dict[layer_name]

        new_dfsch_list = []

        for ifmap_layout, dfsch_idx in self._gen_layer_ifmap_layout(
                layer_name, dfsch_list, options):

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
                dfsch = dfsch_list[dfsch_idx].copy()
                dfsch[layer_name] = t
                new_dfsch_list.append(dfsch)

        # Always pick and keep top n at each layer.
        return sorted(new_dfsch_list, key=lambda dfsch: dfsch.total_cost
                     )[:options.ntops]

    def _gen_layer_ifmap_layout(self, layer_name, dfsch_list, options):
        '''
        Generator to get all the choices of ifmap layout for the layer.

        Return the ifmap layout, and the corresponding NNDataflowScheme index
        in the list.
        '''

        del options

        prev_layer_names, merge_symbol = self.network.prev_layers(layer_name)
        assert prev_layer_names

        def _ofmap_layout(dfsch, pl_name):
            return dfsch[pl_name].ofmap_layout if pl_name is not None \
                    else dfsch.input_layout

        for idx, dfsch in enumerate(dfsch_list):
            # Merge all previous layer ofmap layouts to get the ifmap layout.
            ifmap_layout = _ofmap_layout(dfsch, prev_layer_names[0])
            for pl_name in prev_layer_names[1:]:
                ifmap_layout = ifmap_layout.merge(merge_symbol,
                                                  _ofmap_layout(dfsch, pl_name))

            # Remap dst memory to src memory.
            origin_diff = self.resource.src_data_region().origin \
                    - self.resource.dst_data_region().origin
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

        input_region = self.resource.dst_data_region()

        for part in partition.gen_partition(input_layer, self.batch_size,
                                            input_region.dim, options,
                                            guaranteed=True):
            input_layout = partition.get_ofmap_layout(
                input_layer, self.batch_size, part, input_region)

            yield input_layout

