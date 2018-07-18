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

from collections import defaultdict
import sys

from . import partition
from .cost import Cost
from .data_layout import DataLayout
from .fmap_range import FmapPosition, FmapRange
from .inter_layer_pipeline import InterLayerPipeline
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

        # Inter-layer pipelining.
        self.ilp = InterLayerPipeline(self.network, self.batch_size,
                                      self.resource)
        self.ordered_layer_list = self.ilp.ordered_layer_list()

        # Allowed time overhead due to layer pipelining.
        self.layer_base_time = {}

        # NNDataflowScheme tops.
        # The top schemes are organized by the ending layers, and keeping
        # extended to the end of the network.
        self.nndf_tops = {}

    def schedule_search(self, options):
        '''
        Search the optimized dataflows.
        '''
        # Group the segments by the ending layers.
        segments = defaultdict(list)
        for seg in self.ilp.gen_segment(options):
            if seg not in segments[seg[-1][-1]]:
                segments[seg[-1][-1]].append(seg)

        # Clear and reset.
        self.nndf_tops = {}
        self.layer_base_time = {}

        # Initial input layout.
        self.nndf_tops[None] = []
        for input_layout in self._gen_input_layout(options):
            nndf = NNDataflowScheme(self.network, input_layout)
            self.nndf_tops[None].append(nndf)

        # Schedule layers.
        for layer_name in self.ordered_layer_list:
            if options.verbose:
                sys.stderr.write('-> {}\n'.format(layer_name))
                sys.stderr.flush()

            # The top schemes ending with the current layer.
            tops = []

            # The segments ended with the current layer. Use them to extend the
            # current top schemes.
            for seg in segments.get(layer_name, []):
                if options.verbose:
                    sys.stderr.write('  - {}\n'.format(seg.seg))
                    sys.stderr.flush()
                tops += self._segment_schedule_search(seg, options)

            # Always pick and keep top n.
            tops = sorted(tops, key=NNDataflowScheme.cmp_key)[:options.ntops]

            # Add to the top list.
            assert layer_name not in self.nndf_tops
            self.nndf_tops[layer_name] = tops

        # Final top schemes.
        nndf_tops = self.nndf_tops.get(self.ordered_layer_list[-1], [])
        if not nndf_tops:
            sys.stderr.write('No valid schedule found for {}.\n'
                             .format(self.network.net_name))
        for nndf in nndf_tops:
            assert len(nndf) == len(self.network)

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

    def _segment_schedule_search(self, segment, options):
        '''
        Schedule the given PipelineSegment `segment`.

        Return a list of top NNDataflowScheme instances that include this
        segment. Will NOT update the `nndf_tops` attribute.
        '''

        # Single-layer or multi-layer segment.
        single_layer = len(segment) == 1 and len(segment[0]) == 1
        if single_layer:
            layer = segment[0][0]
            assert layer not in self.layer_base_time
            self.layer_base_time[layer] = float('inf')
            time_limit = float('inf')
        else:
            # Get time limit.
            time_limit = 0
            for ltpl in segment:
                for layer in ltpl:
                    assert layer in self.layer_base_time, \
                            'NNDataflow: single-layer segment must be ' \
                            'yielded before multi-layer segments. ' \
                            'Got {} before {}.'.format(segment, layer)
                    time_limit += self.layer_base_time[layer]
            time_limit *= (1 + options.layer_pipeline_time_ovhd)

        # We take the top schemes that end with the latest previous layer as
        # the initial state.
        first_layer_idx = self.ordered_layer_list.index(segment[0][0])
        if first_layer_idx == 0:
            prev_nndf_tops = self.nndf_tops[None]
        else:
            prev_nndf_tops = self.nndf_tops.get(
                self.ordered_layer_list[first_layer_idx - 1], [])
        if not prev_nndf_tops:
            return []

        # New top schemes.
        nndf_tops = []

        # Allocation.
        allocation = segment.allocation()

        fast_forward = False

        # Explore constraints.
        for constraint, ff_end in segment.gen_constraint():

            # Prune.
            if ff_end:
                # Exit fast forwarding.
                fast_forward = False

            if fast_forward:
                continue

            # Start from the previous top schemes.
            curr_nndf_tops = prev_nndf_tops

            # Spatial scheduling.
            for sp_idx, (ltpl, rtpl, ctpl) \
                    in enumerate(zip(segment, allocation, constraint)):

                # Temporal scheduling.
                for tm_idx, (layer, resource, cstr) \
                        in enumerate(zip(ltpl, rtpl, ctpl)):

                    curr_nndf_tops = self._layer_schedule_search(
                        layer, resource, cstr, sp_idx, tm_idx,
                        curr_nndf_tops, options)

            # Filter by time limit.
            filtered_nndf_tops = [nndf for nndf in curr_nndf_tops
                                  if nndf.last_segment_time() <= time_limit]

            nndf_tops += filtered_nndf_tops

            if not curr_nndf_tops or filtered_nndf_tops:
                # Enter fast forwarding if 1) constraint is infeasible; or 2)
                # already found with less strict constraint.
                fast_forward = True
            else:
                assert curr_nndf_tops and not filtered_nndf_tops
                # This is the case where all schemes are filtered out by the
                # time limit.
                assert all(nndf.last_segment_time() > time_limit
                           for nndf in curr_nndf_tops)
                if all(nndf.last_segment_critical_time() > time_limit
                       for nndf in curr_nndf_tops):
                    # Enter fast forwarding if the critical time does not meet
                    # the time limit, in which case the more strict constraints
                    # will not help.
                    fast_forward = True

        # Always pick and keep top n.
        nndf_tops = sorted(nndf_tops,
                           key=NNDataflowScheme.cmp_key)[:options.ntops]

        # Set base time for scheduling layer individually.
        if single_layer and nndf_tops:
            layer = segment[0][0]
            self.layer_base_time[layer] = nndf_tops[0][layer].total_time

        return nndf_tops

    def _layer_schedule_search(self, layer_name, resource, constraint,
                               spatial_idx, temporal_idx, prev_nndf_tops,
                               options):
        '''
        Schedule the given layer under the given previous top NNDataflowScheme
        instances in 'prev_nndf_tops`.

        `spatial_idx` and `temporal_idx` give the spatial and temporal
        scheduling index in the segment. The segment index is inferred from the
        previous top schemes.

        Return new top NNDataflowScheme instances that include this layer. Will
        NOT update the `nndf_tops` attribute.
        '''
        nndf_tops = []

        layer_sched = self.layer_sched_dict[layer_name]

        for ifmap_layout, prev_nndf in self._gen_layer_ifmap_layout(
                layer_name, prev_nndf_tops):

            segment_idx = prev_nndf.last_seg_idx
            if spatial_idx == 0 and temporal_idx == 0:
                # New segment.
                segment_idx += 1

            sched_seq = (segment_idx, spatial_idx, temporal_idx)

            condition = SchedulingCondition(resource=resource,
                                            constraint=constraint,
                                            ifmap_layout=ifmap_layout,
                                            sched_seq=sched_seq)

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
        return sorted(nndf_tops, key=NNDataflowScheme.cmp_key)[:options.ntops]

    def _gen_layer_ifmap_layout(self, layer_name, prev_nndf_tops):
        '''
        Generate all choices of ifmap layout for the layer, based on the given
        previous top NNDataflowScheme instances in `prev_nndf_tops`.

        Return the ifmap layout, and the corresponding NNDataflowScheme.
        '''
        prevs = self.network.prevs(layer_name)
        assert prevs

        def _ofmap_layout(nndf, prev_layer_name):
            return nndf[prev_layer_name].ofmap_layout if prev_layer_name \
                    else nndf.input_layout

        for nndf in prev_nndf_tops:
            # Merge all previous layer ofmap layouts to get the ifmap layout.
            ifmap_layout = DataLayout.concat(*[_ofmap_layout(nndf, p)
                                               for p in prevs])

            # We already checked the ofmap layout dimension in Scheduling, and
            # the prev/next layer dimensions in Network, so ifmap_layout ==
            # layer == prev_layers == ofmap_layout.

            yield ifmap_layout, nndf

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

        input_region = self.resource.src_data_region

        for part in partition.gen_partition(input_layer, self.batch_size,
                                            input_region.dim, options,
                                            guaranteed=True):
            input_layout = DataLayout(
                frngs=(input_frng,),
                regions=(input_region,),
                parts=(part.projection(input_region, appl2frng=True),))

            yield input_layout

