""" $lic$
Copyright (C) 2016-2019 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

import itertools
import sys

from . import partition
from .cost import Cost
from .data_layout import DataLayout
from .fmap_range import FmapPosition, FmapRange
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
        # Use the same instance for all same layers in order to exploit its
        # scheduling cache.
        layer2sched = {}
        for layer_name in self.network:
            layer = self.network[layer_name]
            sched = layer2sched.get(layer, None)
            if sched is None:
                sched = Scheduling(layer, self.batch_size, self.cost,
                                   self.map_strategy)
                layer2sched[layer] = sched
            self.layer_sched_dict[layer_name] = sched

        # Default compare key function.
        self.cmp_key = lambda nndf: (nndf.total_cost, nndf.total_time)

    def schedule_search(self, options):
        '''
        Search the optimized dataflows.
        '''
        # Set key function.
        if options.opt_goal == 'ed':
            self.cmp_key = lambda nndf: nndf.total_cost * nndf.total_time
        elif options.opt_goal == 'd':
            self.cmp_key = lambda nndf: (nndf.total_time, nndf.total_cost)
        else:
            assert options.opt_goal == 'e'

        # Clear and reset.
        nndf_tops = []

        # Initial input layout.
        for input_layout, ext_layout_dict in self._gen_input_layout(options):
            nndf = NNDataflowScheme(self.network, input_layout, ext_layout_dict)
            nndf_tops.append(nndf)

        # Schedule layers.
        for layer_name in self.network:
            if options.verbose:
                sys.stderr.write('-> {}\n'.format(layer_name))
                sys.stderr.flush()

            nndf_tops = self._layer_schedule_search(
                layer_name, nndf_tops, options)

        # Cache stats.
        cache_hits = 0
        cache_misses = 0
        seen_scheds = set()
        for sched in self.layer_sched_dict.values():
            if sched in seen_scheds:
                continue
            seen_scheds.add(sched)
            h, m = sched.cache_stats()
            cache_hits += h
            cache_misses += m

        return nndf_tops, (cache_hits, cache_misses)

    def _layer_schedule_search(self, layer_name, prev_nndf_tops, options):
        '''
        Schedule the given layer under the given previous top NNDataflowScheme
        instances in 'prev_nndf_tops`.

        Return new top NNDataflowScheme instances that include this layer.
        '''
        nndf_tops = []

        layer_sched = self.layer_sched_dict[layer_name]

        for prev_nndf in prev_nndf_tops:

            ifmap_layout = prev_nndf.fmap_layout(self.network.prevs(layer_name))

            condition = SchedulingCondition(resource=self.resource,
                                            ifmap_layout=ifmap_layout)

            try:
                sched_tops = layer_sched.schedule_search(condition, options)
            except Exception:
                sys.stderr.write('Failed when scheduling layer {}.\n'
                                 .format(layer_name))
                raise

            # Append all the current layer top schedules to all the previous top
            # schedules with the matching fmap layout.
            for t in sched_tops:
                nndf = prev_nndf.copy()
                nndf[layer_name] = t
                nndf_tops.append(nndf)

        # Always pick and keep top n at each layer.
        return sorted(nndf_tops, key=self.cmp_key)[:options.ntops]

    def _gen_input_layout(self, options):
        '''
        Get the input layer layout choices.
        '''
        input_layer = self.network.input_layer()
        input_frng = FmapRange(FmapPosition(b=0, n=0, h=0, w=0),
                               FmapPosition(b=self.batch_size,
                                            n=input_layer.nofm,
                                            h=input_layer.hofm,
                                            w=input_layer.wofm))

        ext_layer_names = self.network.ext_layers()
        ext_layers = [self.network[l] for l in ext_layer_names]
        ext_frngs = [FmapRange(FmapPosition(b=0, n=0, h=0, w=0),
                               FmapPosition(b=self.batch_size,
                                            n=ext_layer.nofm,
                                            h=ext_layer.hofm,
                                            w=ext_layer.wofm))
                     for ext_layer in ext_layers]

        # Input and external layers share the same region.

        input_region = ext_region = self.resource.src_data_region

        for part in partition.gen_partition(input_layer, self.batch_size,
                                            input_region.dim, options,
                                            guaranteed=True):
            input_layout = DataLayout(
                frngs=(input_frng,),
                regions=(input_region,),
                parts=(part.projection(input_region, appl2frng=True),))

            if ext_layers:
                for ext_parts in itertools.product(
                        *[partition.gen_partition(ext_layer, self.batch_size,
                                                  ext_region.dim, options,
                                                  guaranteed=True)
                          for ext_layer in ext_layers]):
                    ext_layout_dict = dict(zip(
                        ext_layer_names,
                        [DataLayout(
                            frngs=(ext_frng,),
                            regions=(ext_region,),
                            parts=(ext_part.projection(ext_region,
                                                       appl2frng=True),))
                         for ext_part, ext_frng in zip(ext_parts, ext_frngs)]))

                    yield input_layout, ext_layout_dict

            else:
                yield input_layout, None

