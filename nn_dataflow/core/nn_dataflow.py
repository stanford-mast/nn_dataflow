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

from . import data_category_enum as de
from . import loop_enum as le
from . import partition
from .. import util
from .cost import Cost
from .inter_layer_pipeline import InterLayerPipeline
from .layer import ConvLayer
from .map_strategy import MapStrategy
from .network import Network
from .nn_dataflow_scheme import NNDataflowScheme
from .resource import Resource
from .scheduling import SchedulingCondition, Scheduling
from .scheduling_constraint import SchedulingConstraint

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

        # The cmp function to sort and pick the top NNDataflowScheme instances.
        self.cmp_func = NNDataflowScheme.compare_cost_with_time_overhead

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
            segments[seg[-1][-1]].append(seg)

        # Clear and reset.
        self.nndf_tops = {}

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
                    sys.stderr.write('  - {}\n'.format(seg))
                    sys.stderr.flush()
                tops += self._segment_schedule_search(seg, options)

            # Always pick and keep top n.
            tops = sorted(tops, cmp=self.cmp_func)[:options.ntops]

            # Add to the top list.
            assert layer_name not in self.nndf_tops
            self.nndf_tops[layer_name] = tops

        # Final top schemes.
        try:
            tops = self.nndf_tops[self.ordered_layer_list[-1]]
        except KeyError:
            sys.stderr.write('No valid schedule found for {}.\n'
                             .format(self.network.net_name))
            tops = []
        for nndf in tops:
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

        return tops, (cache_hits, cache_misses)

    def _segment_schedule_search(self, segment, options):
        '''
        Schedule the given PipelineSegment `segment`.

        Return a list of top NNDataflowScheme instances that include this
        segment. Will NOT update the `nndf_tops` attribute.
        '''

        allocation = segment.allocation()
        assert allocation

        # We take the top schemes that end with the latest previous layer as
        # the initial state.
        first_layer_idx = self.ordered_layer_list.index(segment[0][0])
        if first_layer_idx == 0:
            prev_nndf_tops = self.nndf_tops[None]
        else:
            try:
                prev_nndf_tops = self.nndf_tops[
                    self.ordered_layer_list[first_layer_idx - 1]]
            except KeyError:
                return []

        def _do_chedule_search(*conditions):
            ''' Schedule the segment under the given pipelining conditions. '''

            nndf_tops = prev_nndf_tops

            constraint_list = self._gen_segment_scheduling_constraint(
                segment, *conditions)

            # Spatial scheduling.
            for sp_idx, (ltpl, rtpl, ctpl) \
                    in enumerate(zip(segment, allocation, constraint_list)):

                # Temporal scheduling.
                for tm_idx, (layer, resource, constraint) \
                        in enumerate(zip(ltpl, rtpl, ctpl)):

                    nndf_tops = self._layer_schedule_search(
                        layer, resource, constraint, sp_idx, tm_idx,
                        nndf_tops, options)

            return nndf_tops

        # Decide the fmap temporal partitioning factor candidates, which should
        # be approximately dividable by hofm of the layers.
        fmap_tpart_cands = []
        seg_layer_hofm_list = [self.network[l].hofm
                               for ltpl in segment for l in ltpl]
        for f in range(1, min(seg_layer_hofm_list) + 1):
            if all(util.approx_dividable(hofm, f, overhead=0.5)
                   for hofm in seg_layer_hofm_list):
                fmap_tpart_cands.append(f)
        assert fmap_tpart_cands[0] == 1

        # Decide the top BAT factor, sorted from largest to smallest.
        def _gen_top_tb_cands(fmap_tpart):
            if len(segment) == 1:
                return [None]
            return sorted((t for t, _ in util.factorize(
                self.batch_size * fmap_tpart, 2)), reverse=True)

        # Decide the fully buffered starting data category.
        fb_dce_cands = [de.IFM, de.OFM] if len(segment) > 1 else [None]

        # New top schemes.
        nndf_tops = []

        # Explore.
        for fmap_tpart in fmap_tpart_cands:

            for fb_dce in fb_dce_cands:

                tops = []

                for top_tb in _gen_top_tb_cands(fmap_tpart):

                    tops += _do_chedule_search(fmap_tpart, fb_dce, top_tb)

                    if not tops:
                        # If not found for larger top tb, smaller top tb is
                        # also invalid.
                        break

                    nndf_tops += tops

            if nndf_tops:
                # If found, stop using larger fmap temporal partitioning.
                break

        # Always pick and keep top n.
        nndf_tops = sorted(nndf_tops, cmp=self.cmp_func)[:options.ntops]

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

        layer_sched = self.layer_sched_dict[layer_name]

        nndf_tops = []

        for ifmap_layout, prev_nndf in self._gen_layer_ifmap_layout(
                layer_name, prev_nndf_tops):

            segment_idx = prev_nndf.last_sched_seq[0]
            if spatial_idx == 0 and temporal_idx == 0:
                # New segment.
                segment_idx += 1

            sched_seq = (segment_idx, spatial_idx, temporal_idx)

            condition = SchedulingCondition(resource=resource,
                                            constraint=constraint,
                                            ifmap_layout=ifmap_layout,
                                            sched_seq=sched_seq)

            try:
                tops = layer_sched.schedule_search(condition, options)
            except Exception:
                sys.stderr.write('Failed when scheduling layer {}.\n'
                                 .format(layer_name))
                raise

            # Append all the current layer top schedules to all the previous top
            # schedules with the matching fmap layout.
            for t in tops:
                nndf = prev_nndf.copy()
                nndf[layer_name] = t
                nndf_tops.append(nndf)

        # Always pick and keep top n at each layer.
        return sorted(nndf_tops, cmp=self.cmp_func)[:options.ntops]

    def _gen_layer_ifmap_layout(self, layer_name, prev_nndf_tops):
        '''
        Generate all choices of ifmap layout for the layer, based on the given
        previous top NNDataflowScheme instances in `prev_nndf_tops`.

        Return the ifmap layout, and the corresponding NNDataflowScheme.
        '''

        prev_layer_names, merge_symbol = self.network.prev_layers(layer_name)
        assert prev_layer_names

        def _ofmap_layout(nndf, pl_name):
            ofmap_layout = nndf[pl_name].ofmap_layout \
                    if pl_name is not None else nndf.input_layout
            if ofmap_layout.is_in_region(self.resource.dst_data_region()):
                # Remap dst memory to src memory.
                origin_diff = self.resource.src_data_region().origin \
                        - self.resource.dst_data_region().origin
                ofmap_layout = ofmap_layout.view(origin_diff=origin_diff)
            return ofmap_layout

        for nndf in prev_nndf_tops:
            # Merge all previous layer ofmap layouts to get the ifmap layout.
            it = iter(prev_layer_names)
            ifmap_layout = _ofmap_layout(nndf, next(it))
            for pl_name in it:
                ifmap_layout = ifmap_layout.merge(
                    merge_symbol, _ofmap_layout(nndf, pl_name))

            # We already checked the ofmap layout dimension in Scheduling, and
            # the prev/next layer dimensions in Network, so ifmap_layout ==
            # layer == prev_layers == ofmap_layout.

            yield ifmap_layout, nndf

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

    def _gen_segment_scheduling_constraint(self, segment, *conditions):
        '''
        Generate SchedulingConstraint instances for each layer in the segment
        with mixed spatial and temporal scheduling, based on the given
        conditions.

        Return a list (for spatial scheduling) of sub-lists (for temporal
        scheduling), in the similar format of segment and allocation.

        Conditions include:
        - fmap temporal partitioning factor, for all layers in the segment.
        - fully buffered data category (IFM or OFM) of the starting layer.
          - between spatial scheduling:
            - if there is single spatial scheduling, it starts with no
              requirement on fully buffering.
            - if there is multiple spatial scheduling, the first one can start
              with either data category (IFM or OFM), and the afterward ones
              inherit the final requirement of their previous ones.
          - with all temporal scheduling in the same spatial scheduling:
            - all local-region layers are merged with their previous CONV
              layers, composing CONV layer groups.
            - the fully buffering requirement only applies to the first (IFM)
              or the last (OFM) layer in the group. Layers in between do not
              need to fully buffer (can stream instead).
            - if there are multiple CONV layer groups, the first group inherits
              the fully buffered data category from the previous spatial
              scheduling; all the groups need to fully buffer both IFM and OFM,
              except that the first group does not need to fully buffer IFM
              (unless inherit), and the last group does not need to fully
              buffer OFM.
            - if there is a single CONV layer group, the fully buffered data
              category alternates.
            - if there is no CONV layer, no requirement on fully buffering.
        - top BAT loop factor. With > 1 spatial scheduling, all layers must
          share the same factor; with single spatial scheduling, each temporal
          scheduled layers can use different factors.
        '''

        fmap_tpart, fb_dce, top_tb = conditions

        def _fb_constrained_lpe(dce):
            '''
            The constrained LoopEnum for the fully buffered data category.
            '''
            return {de.IFM: le.IFM, de.OFM: le.OFM, None: None}[dce]

        def _fb_alter(dce):
            '''
            Alternate the fully buffer data category.
            '''
            return {de.IFM: de.OFM, de.OFM: de.IFM, None: None}[dce]

        # Spatial scheduling.
        cnt_spat = len(segment)

        # Single spatial scheduling does not have requirements for top tb and
        # fully buffering.
        assert cnt_spat > 1 or top_tb is None
        assert cnt_spat > 1 or fb_dce is None

        top_bl_lpe = le.BAT if top_tb is not None else None

        top_bl_t_list = []

        for ltpl in segment:

            # Temporal scheduling.
            cnt_temp = len(ltpl)

            # Find CONV layer groups, representing by the CONV layer index.
            conv_idx_list = []
            for tm_idx, layer in enumerate(ltpl):
                if isinstance(self.network[layer], ConvLayer):
                    conv_idx_list.append(tm_idx)
            assert all(p < n for p, n in zip(conv_idx_list, conv_idx_list[1:]))

            # Constraints for top loop factors.
            ttpl = [[None] * le.NUM for _ in range(cnt_temp)]
            # Top BAT factor.
            for top_bl_t in ttpl:
                top_bl_t[le.BAT] = top_tb

            if not conv_idx_list:
                # No CONV layers.

                # All layers have no requirements.
                pass

            else:
                # Get the begin and end of each group.
                groups = [(beg, beg2 - 1) for beg, beg2
                          in zip(conv_idx_list,
                                 conv_idx_list[1:] + [cnt_temp])]
                assert len(groups) == len(conv_idx_list)

                # All groups buffers both IFM and OFM.
                for beg, end in groups:
                    ttpl[beg][le.IFM] = 1
                    ttpl[end][le.OFM] = 1

                # Cancel the requirement of the first and last group.
                ttpl[groups[0][0]][le.IFM] = None
                ttpl[groups[-1][-1]][le.OFM] = None

                # The first group inherits previous spatial schedule.
                if fb_dce == de.IFM:
                    ttpl[groups[0][0]][le.IFM] = 1

                    if groups[0][0] > 0:
                        # If there are layers before the first group, and the
                        # first group fully buffers IFM, this semi-group needs
                        # to fully buffer OFM (in the last layer only).
                        ttpl[groups[0][0] - 1][le.OFM] = 1

                if len(conv_idx_list) == 1:
                    # Alternate fully-buffered data category.
                    fb_dce = _fb_alter(fb_dce)
                else:
                    # End with fully buffering OFM for the next spatial
                    # schedule.
                    fb_dce = de.OFM

            top_bl_t_list.append(ttpl)

        # IFM of the first layer and OFM of the last layer can go to DRAM.
        top_bl_t_list[0][0][le.IFM] = None
        top_bl_t_list[-1][-1][le.OFM] = None

        # Make SchedulingConstraint instances.
        constraint_list = [[SchedulingConstraint(top_bl_t=tuple(top_bl_t),
                                                 top_bl_lpe=top_bl_lpe,
                                                 fmap_tpart=fmap_tpart)
                            for top_bl_t in ttpl]
                           for ttpl in top_bl_t_list]

        return constraint_list

