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

class NNDataflow():
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

        # NNDataflowScheme tops.
        # The top schemes are organized by the ending layers, and keeping
        # extended to the end of the network.
        self.nndf_tops = {}

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

        # Group the segments by the ending layers.
        segments = defaultdict(list)
        for seg in self.ilp.gen_segment(options):
            if seg not in segments[seg[-1][-1]]:
                segments[seg[-1][-1]].append(seg)

        # Clear and reset.
        self.nndf_tops = {}

        # Initial input layout.
        self.nndf_tops[None] = []
        for input_layout, ext_layout_dict in self._gen_input_layout(options):
            nndf = NNDataflowScheme(self.network, input_layout, ext_layout_dict)
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
            for seg in segments[layer_name]:
                if options.verbose:
                    sys.stderr.write('  - {}\n'.format(seg.seg))
                    sys.stderr.flush()
                tops += self._segment_schedule_search(seg, options)

            # Always pick and keep top n.
            tops = sorted(tops, key=self.cmp_key)[:options.ntops]

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

        Return new top NNDataflowScheme instances that include this segment.
        Will NOT update the `nndf_tops` attribute.
        '''
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

        # Forwarding data regions. Map a spatial index to the forwarding region.
        fwd_data_region_dict = {}
        for sh_list in segment.ifm_fwd_dict.values():
            # A list of spatial indices that share the same ifmaps.
            r = allocation[sh_list[0].sp_idx][sh_list[0].tm_idx].proc_region
            for idx in sh_list[1:]:
                fwd_data_region_dict[idx] = r
        for fwd_src, fwd_dst_list in segment.ofm_fwd_dict.items():
            # Ofmaps forwarded to neighbors.
            r = allocation[fwd_src.sp_idx][fwd_src.tm_idx].proc_region
            for idx in fwd_dst_list:
                fwd_data_region_dict[idx] = r

        # Max allowed time overhead for segment timing.
        max_time_ovhd = options.layer_pipeline_time_ovhd

        # Cost hint Pareto-optimal frontier.
        frontier = set()

        # Explore constraints.
        for constraint, hints in segment.gen_constraint(max_time_ovhd):

            # Filter out off-frontier constraints.
            if any(all(h >= fh for h, fh in zip(hints, fhints))
                   for fhints in frontier):
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
                        fwd_data_region_dict.get((sp_idx, tm_idx)),
                        curr_nndf_tops, options)

            # Filter by time limit.
            seg_nndf_tops = [nndf for nndf in curr_nndf_tops
                             if all(timing.time_overhead <= max_time_ovhd
                                    for timing in nndf.segment_timing_list)]

            # Add to frontier.
            if seg_nndf_tops:
                frontier.add(hints)

            nndf_tops += seg_nndf_tops

        # Always pick and keep top n.
        return sorted(nndf_tops, key=self.cmp_key)[:options.ntops]

    def _layer_schedule_search(self, layer_name, resource, constraint,
                               spatial_idx, temporal_idx, fwd_data_region,
                               prev_nndf_tops, options):
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

        for prev_nndf in prev_nndf_tops:

            ifmap_layout = prev_nndf.fmap_layout(self.network.prevs(layer_name))

            if fwd_data_region is not None:
                # Remap source data regions to the forwarding region.
                ifmap_layout = DataLayout(
                    frngs=ifmap_layout.frngs,
                    regions=(fwd_data_region,) * len(ifmap_layout.frngs),
                    parts=tuple(p.projection(fwd_data_region, appl2frng=True)
                                for p in ifmap_layout.parts))

            segment_idx = prev_nndf.last_seg_idx
            if spatial_idx == 0 and temporal_idx == 0:
                # New segment.
                segment_idx += 1

            sched_seq = (segment_idx, spatial_idx, temporal_idx)

            constraint.update_by_prev(prev_nndf)

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

            ext_layout_dict = dict(zip(
                ext_layer_names,
                [DataLayout(
                    frngs=(ext_frng,),
                    regions=(ext_region,),
                    parts=(part.projection(ext_region, appl2frng=True),))
                 for ext_frng in ext_frngs])) if ext_layers else None

            yield input_layout, ext_layout_dict
