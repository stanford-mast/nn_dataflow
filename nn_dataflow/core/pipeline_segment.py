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

from collections import namedtuple

from . import loop_enum as le
from .. import util
from .layer import ConvLayer
from .network import Network
from .resource import Resource
from .scheduling_constraint import SchedulingConstraintLayerPipeline as Cstr

class PipelineSegment(object):
    '''
    Inter-layer pipeline segment.

    Segment is a two-level layer hierarchy, where the first level is spatially
    scheduled and the second level is temporally scheduled.
    '''

    # Scheduling index in the segment, as a tuple of spatial and temporal
    # scheduling indices.
    SchedIndex = namedtuple('SchedIndex', ['sp_idx', 'tm_idx'])

    def __init__(self, seg, network, batch_size, resource, max_util_drop=0.05):
        if not isinstance(seg, tuple):
            raise TypeError('PipelineSegment: seg must be a tuple.')
        for ltpl in seg:
            if not isinstance(ltpl, tuple):
                raise TypeError('PipelineSegment: seg must be a tuple '
                                'of sub-tuples.')

        if not isinstance(network, Network):
            raise TypeError('PipelineSegment: network must be '
                            'a Network instance.')
        if not isinstance(resource, Resource):
            raise TypeError('PipelineSegment: resource must be '
                            'a Resource instance.')

        self.seg = seg
        self.network = network
        self.batch_size = batch_size
        self.resource = resource
        self.max_util_drop = max_util_drop

        self.valid = self._init_deps()
        if not self.valid:
            return

        # Resource allocation.
        self.valid = self._alloc_resource(max_util_drop=max_util_drop)
        if not self.valid:
            return

    def allocation(self):
        '''
        Get resource allocation, as a tuple of sub-tuples corresponding to the
        layers in the segment.
        '''
        if not self.valid:
            return None
        return self.alloc

    def gen_constraint(self):
        '''
        Generate scheduling constraint for the segment, as a tuple of
        sub-tuples of SchedulingConstraint instances, corresponding to the
        layers in the segment.

        Yield the segment constraint tuple, and information for pruning and
        early termination. The information includes:

        - ff_end: if True, starting from (including) the current constraint,
          the constraints will have NO strictness relation with the previous
          ones. Between two ff_end points, the constraints are guaranteed to be
          increasingly strict. If previously we have encountered an infeasible
          constraint, we can fast forward until a ff_end point.

        Rules for constraints.

        1. Top BAT loop factor.

        With a single spatial scheduling, there is no constraint on top BAT
        loop factor. Otherwise all layers must share the same factor.

        2. Fully buffered data category (ifmaps or ofmaps).

        Only CONV layers require fully buffered data. Local-region layers
        process data in a streaming manner. We group each CONV layer, and all
        local-region layers immediately following it within the same spatial
        scheduling, into a group. The fully buffering requirements only apply
        to the first layer (ifmaps) or the last layer (ofmaps) in the group.
        Layers in between do not need to fully buffer.

        For a group G in the segment,

        - (initial) if G is both first spatial and temporal scheduled, it can
          choose whether to fully buffer ofmaps or not. This is a configuration
          to explore.

        - (between spatial) if G has a source from G' in another spatial
          scheduling, the source, as a neighbor source dependency, must be the
          last temporal scheduled in both G' and that spatial scheduling,
          - If G' fully buffers only ofmaps, G fully buffers ifmaps.
          - Otherwise, make G' fully buffer ifmaps, and G fully buffer ofmaps
            (we bias this case as it can reduce the pipeline filling delay).

        - (between temporal) if G has a source from G' in the same spatial
          scheduling, the source, as a local source dependency, must be
          immediately before G. G' must fully buffer ofmaps, and G must fully
          buffer ifmaps.
        '''

        # Top BAT factors, sorted from smallest to largest.
        def _top_tb_cands():
            cands = [None] if len(self.seg) == 1 else \
                (t for t, _ in util.factorize(self.batch_size, 2))
            return sorted(cands)

        # Start with fully buffer ofmaps or not.
        sfbo_cands = [False] if len(self.seg) == 1 else [True, False]

        # Pruning info.
        ff_end = False
        constraint_set = set()

        for sfbo in sfbo_cands:

            for top_tb in _top_tb_cands():

                conditions = (top_tb, sfbo)

                constraint = self._make_constraint(*conditions)

                if constraint and constraint not in constraint_set:

                    yield constraint, ff_end

                    # Reset info until next set.
                    ff_end = False

                    constraint_set.add(constraint)

            # Smaller top tb factors are more strict than larger ones.
            ff_end = True

    def __getitem__(self, index):
        return self.seg[index]

    def __iter__(self):
        return self.seg.__iter__()

    def __len__(self):
        return len(self.seg)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # pylint: disable=protected-access
            return self._key_attrs() == other._key_attrs()
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(self._key_attrs()))

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'seg={}'.format(repr(self.seg)),
                'network={}'.format(repr(self.network)),
                'batch_size={}'.format(repr(self.batch_size)),
                'resource={}'.format(repr(self.resource)),
                'max_util_drop={}'.format(repr(self.max_util_drop))]))

    def _key_attrs(self):
        ''' Used for comparison. '''
        return (self.seg, self.network, self.batch_size, self.resource,
                self.max_util_drop)

    def _init_deps(self):
        '''
        Initialize the dependency relationship of the layers in the segment as
        a mapping of the scheduling indices, and check validation. Return
        whether the segment is valid to schedule.

        We categorize dependencies to 3 categories:
        - local: with the same spatial index but different temporal indices;
        - neighbor: with different spatial indices but in the same segment;
        - memory: in different segments, from/to memory.

        The values of the src/dst dicts are tuples of indices of the neighbor
        dependencies. A layer can have at most one neighbor source (must be a
        last temporal scheduled layer), but may have multiple neighbor
        destinations (could be temporal scheduled in the middle). Also, all
        layers with the same spatial index can have at most one neighbor
        source.

        Special index `None` means memory dependency, i.e., from/to memory.
        Memory dependencies and neighbor dependencies are mutual exclusive,
        based on the segment generation rule (see InterLayerPipeline).

        Local dependencies are omitted, as by default each layer has its
        immediately previous layer as local source and immediately next layer
        as local destination.
        '''

        self.src_dict = [[None for _ in ltpl] for ltpl in self.seg]
        self.dst_dict = [[None for _ in ltpl] for ltpl in self.seg]

        # Mapping from layer to spatial/temporal indices in the segment.
        layer2idx = {layer: PipelineSegment.SchedIndex(sp_idx, tm_idx)
                     for sp_idx, ltpl in enumerate(self.seg)
                     for tm_idx, layer in enumerate(ltpl)}

        for sp_idx, ltpl in enumerate(self.seg):

            cnt_nbr_src = 0

            for tm_idx, layer in enumerate(ltpl):

                assert layer2idx[layer] == (sp_idx, tm_idx)

                # Sources.
                src = tuple()

                prev_layers, _ = self.network.prev_layers(layer)
                assert all(l not in layer2idx or layer2idx[l] < layer2idx[layer]
                           for l in prev_layers)
                mem_src = [l for l in prev_layers if l not in layer2idx]
                lcl_src = [l for l in prev_layers if l not in mem_src
                           and layer2idx[l].sp_idx == sp_idx]
                nbr_src = [l for l in prev_layers if l not in mem_src + lcl_src]

                # Ensure single local source to be the immediately previous.
                # Check at the destination so here are assertions.
                if not lcl_src:
                    assert tm_idx == 0
                else:
                    assert len(lcl_src) == 1 \
                            and layer2idx[lcl_src[0]].tm_idx == tm_idx - 1

                # Mutual exclusive.
                assert not mem_src or not nbr_src

                if mem_src:
                    # Memory source.
                    src += (None,)
                if nbr_src:
                    # Neighbor source.
                    # Single neighbor source to be the last temporal scheduled.
                    assert len(nbr_src) == 1
                    prev_idx = layer2idx[nbr_src[0]]
                    assert prev_idx.tm_idx == len(self.seg[prev_idx.sp_idx]) - 1
                    # Single neighbor source across this spatial scheduling.
                    cnt_nbr_src += 1
                    assert cnt_nbr_src <= 1
                    src += (prev_idx,)

                # Destinations.
                dst = tuple()

                next_layers = self.network.next_layers(layer)
                assert all(l not in layer2idx or layer2idx[l] > layer2idx[layer]
                           for l in next_layers)
                mem_dst = [l for l in next_layers if l not in layer2idx]
                lcl_dst = [l for l in next_layers if l not in mem_dst
                           and layer2idx[l].sp_idx == sp_idx]
                nbr_dst = [l for l in next_layers if l not in mem_dst + lcl_dst]

                # Ensure single local destination to be the immediate next.
                if not lcl_dst:
                    if tm_idx != len(ltpl) - 1:
                        # Not utilize local data, sub-optimal.
                        return False
                else:
                    if len(lcl_dst) != 1 \
                            or layer2idx[lcl_dst[0]].tm_idx != tm_idx + 1:
                        # Local data will not be available if not adjacent.
                        return False

                # Mutual exclusive.
                assert not mem_dst or not nbr_dst

                if mem_dst:
                    # Memory destination.
                    dst += (None,)
                if nbr_dst:
                    # Neighbor destinations.
                    # This layer is the last temporal scheduled.
                    assert tm_idx == len(ltpl) - 1
                    dst += tuple(nbr_dst)

                self.src_dict[sp_idx][tm_idx] = src
                self.dst_dict[sp_idx][tm_idx] = dst

        return True

    def _alloc_resource(self, max_util_drop=0.05):
        '''
        Decide the resource allocation. Return whether the allocation succeeds.

        `max_util_drop` specifies the maximum utilization drop due to mismatch
        throughput between layers.
        '''

        self.alloc = tuple()

        # Allocate processing subregions.
        subregions = self._alloc_proc(max_util_drop=max_util_drop)
        if not subregions:
            return False

        for sp_idx, ltpl in enumerate(self.seg):

            # Resource for the subregion.
            rtpl = tuple()

            for tm_idx, _ in enumerate(ltpl):

                # Processing region.
                proc_region = subregions[sp_idx]

                # Data source.
                src = self.src_dict[sp_idx][tm_idx]
                if None in src:
                    # Data source is memory.
                    assert src == (None,)
                    src_data_region = self.resource.src_data_region()
                elif src:
                    # Data source is neighbor.
                    assert len(src) == 1
                    src_data_region = subregions[src[0].sp_idx]
                else:
                    # Data source is all local.
                    src_data_region = proc_region

                # Data destination.
                dst = self.dst_dict[sp_idx][tm_idx]
                if None in dst:
                    # Data destination is memory.
                    assert dst == (None,)
                    dst_data_region = self.resource.dst_data_region()
                elif dst:
                    # Data destinations are neighbors.
                    # Put data in local. The next layers will fetch.
                    dst_data_region = proc_region
                else:
                    # Data destination is all local.
                    dst_data_region = proc_region

                # Make resource.
                # Note that DRAM bandwidth is not split here. We optimistically
                # assume each layer can use the full DRAM bandwidth at
                # different time. We adjust this assumption when calculating
                # the segment timing.
                rtpl += (Resource(proc_region=proc_region,
                                  data_regions=(src_data_region,
                                                dst_data_region),
                                  dim_array=self.resource.dim_array,
                                  size_gbuf=self.resource.size_gbuf,
                                  size_regf=self.resource.size_regf,
                                  array_bus_width=self.resource.array_bus_width,
                                  dram_bandwidth=self.resource.dram_bandwidth,
                                 ),)

            assert len(rtpl) == len(ltpl)
            self.alloc += (rtpl,)
        assert len(self.alloc) == len(self.seg)

        return True

    def _alloc_proc(self, max_util_drop=0.05):
        '''
        Allocate processing subregions for the segment.

        Return a list of processing subregions corresponding to the first-level
        (spatial scheduled) layers in the segment. Return None if allocation
        failed.

        `max_util_drop` specifies the maximum utilization drop due to mismatch
        throughput between layers.
        '''

        # Spatial allocation.
        proc_region = self.resource.proc_region
        dim_nodes = proc_region.dim
        total_nodes = dim_nodes.size()

        # Number of operations of each spatial allocation.
        ops = [sum(self.network[l].total_ops() for l in ltpl)
               for ltpl in self.seg]

        # Enforce a common factor among the numbers of nodes allocated to all
        # vertices in the segment. Such common factor is likely to be the
        # common height of the vertex node regions.
        common_factor_list = [cf for cf, _ in util.factorize(dim_nodes.h, 2)]

        for cf in sorted(common_factor_list, reverse=True):
            # Pick the largest common factor within the utilization constraint.

            # Number of nodes of each vertex should be approximate to the
            # number of ops of the vertex.
            nodes_raw = [o * 1. / sum(ops) * total_nodes for o in ops]

            # Round to the common factor multiples.
            assert total_nodes % cf == 0
            nodes = [int(round(nr / cf)) * cf for nr in nodes_raw]
            # Fix margin.
            while sum(nodes) != total_nodes:
                diff = [n - nr for n, nr in zip(nodes, nodes_raw)]
                if sum(nodes) > total_nodes:
                    # Decrease the nodes for the vertex with the maximum
                    # positive difference.
                    idx, _ = max(enumerate(diff), key=lambda tpl: tpl[1])
                    nodes[idx] -= cf
                else:
                    # Increase the nodes for the vertex with the minimum
                    # negative difference.
                    idx, _ = min(enumerate(diff), key=lambda tpl: tpl[1])
                    nodes[idx] += cf

            if 0 in nodes:
                continue

            # Utilization.
            time = max(o * 1. / n for o, n in zip(ops, nodes))
            utilization = sum(ops) / time / sum(nodes)
            assert utilization < 1 + 1e-6

            if utilization >= 1 - max_util_drop:
                # Found
                break

        else:
            # Not found.
            return None

        # Allocate in the processing region according to the number of nodes.
        subregions = proc_region.allocate(nodes)
        assert subregions
        assert len(subregions) == len(self.seg)
        if len(subregions) == 1:
            assert subregions[0] == proc_region

        return subregions

    def _make_constraint(self, *conditions):
        '''
        Make scheduling constraint for the segment under the given conditions.

        See gen_constraint() for the rules.
        '''

        top_tb, sfbo = conditions

        # Single spatial scheduling does not have requirements for top tb and
        # starting fully buffering.
        assert len(self.seg) > 1 or top_tb is None
        assert len(self.seg) > 1 or not sfbo

        if top_tb is None:
            top_tb = 0

        # Scheduling indices for the last CONV layer.
        last_idx = PipelineSegment.SchedIndex(-1, 0)
        # Whether to fully buffer ofmaps for the last group (defer applying to
        # the last layer of the group).
        last_fbo = False

        # Top loop factor for each layer.
        top_bl_t_list = [[[0] * le.NUM for _ in ltpl] for ltpl in self.seg]

        for sp_idx, ltpl in enumerate(self.seg):

            for tm_idx, layer in enumerate(ltpl):

                top_bl_t_list[sp_idx][tm_idx][le.BAT] = top_tb

                if isinstance(self.network[layer], ConvLayer):

                    # Defer applying fully buffer ofmaps to the last group.
                    if last_fbo:
                        if last_idx.sp_idx == sp_idx:
                            # Last group is in the same spatial scheduling.
                            # Apply to immediate previous layer.
                            assert tm_idx >= 1
                            top_bl_t_list[sp_idx][tm_idx - 1][le.OFM] = 1
                        else:
                            # Last group is in a previous spatial scheduling.
                            # Apply to its last temporal scheduled layer.
                            top_bl_t_list[last_idx.sp_idx][-1][le.OFM] = 1
                    # Reset
                    last_fbo = False

                    if last_idx.sp_idx < 0:
                        # Initial rule.
                        last_fbo = sfbo

                    src_deps = self.src_dict[sp_idx][tm_idx]

                    if any(s is not None for s in src_deps):
                        # Between-spatial rule.
                        assert len(src_deps) == 1
                        src_sp_idx, src_tm_idx = src_deps[0]
                        assert src_sp_idx < sp_idx
                        src_top_bl_t = top_bl_t_list[src_sp_idx][src_tm_idx]
                        if src_top_bl_t[le.OFM] == 1 \
                                and src_top_bl_t[le.IFM] != 1:
                            # When neighbor source only fully buffers ofmaps.
                            # Make this group fully buffers ifmaps.
                            top_bl_t_list[sp_idx][tm_idx][le.IFM] = 1
                        else:
                            # Make neighbor source fully buffer ifmaps.
                            src_top_bl_t[le.IFM] = 1
                            # Make this group fully buffer ofmaps.
                            last_fbo = True

                    if last_idx.sp_idx == sp_idx:
                        # Between-temporal rule.
                        # Make last group fully buffer ofmaps.
                        assert tm_idx >= 1
                        top_bl_t_list[sp_idx][tm_idx - 1][le.OFM] = 1
                        # Make this group fully buffer ifmaps.
                        top_bl_t_list[sp_idx][tm_idx][le.IFM] = 1

                    last_idx = PipelineSegment.SchedIndex(sp_idx, tm_idx)

        # The last group ofmaps go to memory, do not need to fully buffer.
        # Ignore final last_fbo value.

        # Make SchedulingConstraint instances.
        constraint = tuple(tuple(Cstr(topbat=top_bl_t[le.BAT],
                                      topifm=top_bl_t[le.IFM],
                                      topofm=top_bl_t[le.OFM])
                                 for top_bl_t in tlst)
                           for tlst in top_bl_t_list)

        # Check whether the constraint is possible to realize.
        for ltpl, rtpl, ctpl in zip(self.seg, self.alloc, constraint):
            for layer, resource, cstr in zip(ltpl, rtpl, ctpl):

                # Required GBUF size.
                req_size_gbuf = 0
                if cstr.topifm == 1:
                    # Fully buffer ifmaps.
                    req_size_gbuf += self.network[layer].total_ifmap_size()
                if cstr.topifm == 1:
                    # Fully buffer ofmaps.
                    req_size_gbuf += self.network[layer].total_ofmap_size()

                if req_size_gbuf > \
                        resource.size_gbuf * resource.proc_region.dim.size():
                    return None

        return constraint

