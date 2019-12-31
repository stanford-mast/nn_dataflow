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

from collections import namedtuple, OrderedDict, Counter
import itertools

from sympy import symbols
from sympy import Basic as symbasic
from sympy import Eq as symeq
from sympy.core.containers import Tuple as symtuple
from sympy.functions.elementary.piecewise import Piecewise as sympiecewise

from .. import util
from .layer import ConvLayer
from .network import Network
from .resource import Resource
from .scheduling_constraint import SchedulingConstraintLayerPipeline as Cstr

class PipelineSegment():
    '''
    Inter-layer pipeline segment.

    Segment is a two-level layer hierarchy, where the first level is spatially
    scheduled and the second level is temporally scheduled.
    '''

    # pylint: disable=too-many-instance-attributes

    # Scheduling index in the segment, as a tuple of spatial and temporal
    # scheduling indices.
    SchedIndex = namedtuple('SchedIndex', ['sp_idx', 'tm_idx'])

    def __init__(self, seg, network, batch_size, resource, max_util_drop=0.05,
                 with_opt=True):
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
        self.with_opt = with_opt

        self.valid = self._init_deps()
        if not self.valid:
            return

        # Resource allocation.
        self.valid = self._alloc_resource(max_util_drop=max_util_drop)
        if not self.valid:
            return

        # Scheduling constraints.
        self.valid = self._init_sym_cstrs()
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

    def gen_constraint(self, max_time_overhead=float('inf')):
        '''
        Generate scheduling constraint for the segment, as a tuple of
        sub-tuples of SchedulingConstraint instances, corresponding to the
        layers in the segment.

        Yield the segment constraint tuple, and hints for pruning.

        Pruning hints are the top-level loop blocking factors. Smaller hints
        indicate better (lower) cost, and larger hints indicate better segment
        timing (with lower time overhead). Constraints with smaller hints are
        generated before those with larger hints. So if a constraint results in
        a valid scheduling, the later ones with all hints larger than its can
        be pruned.
        '''
        syms = self.cstr_symvals.keys()
        vals = self.cstr_symvals.values()
        assert syms and vals

        # Sort from small to large.
        # This is not a strict ordering, but we guarantee that if all values in
        # hint A are larger than the corresponding values in hint B, A will be
        # generated after B.
        vals = [sorted(v) for v in vals]
        syms = list(syms)

        if self.cstr_topbat_idx is not None:
            # Tovhd =  (1 + 1/to + 1 + 1/to + ...) / tb
            #       >= (1 + 1 + ...) / tb = num_sp_fbs / tb
            min_topbat = 1. * self.cstr_num_sp_fbs / max_time_overhead
            pos = self.cstr_topbat_idx
            vals[pos] = [t for t in vals[pos] if t >= min_topbat]

        for valp in itertools.product(*vals):

            constraint = tuple()

            for atpl in self._subs_symargs(self.cstr_symargs,
                                           tuple(zip(syms, valp))):
                ctpl = tuple()
                for a in atpl:
                    # Construct kwargs, adjust the types of the values.
                    kwargs = {}
                    kwargs['topbat'] = int(a.get('topbat', 0))
                    kwargs['fbifm'] = bool(a.get('fbifm', False))
                    if not kwargs['fbifm']:
                        kwargs['topifm'] = int(a.get('topifm', 0))
                    kwargs['fbofm'] = bool(a.get('fbofm', False))
                    if not kwargs['fbofm']:
                        kwargs['topofm'] = int(a.get('topofm', 0))
                    kwargs['update_dict'] = a.get('update_dict')

                    c = Cstr(**kwargs)
                    ctpl += (c,)
                constraint += (ctpl,)

            if None in valp:
                assert len(valp) == 1
                hints = (1,)
            else:
                hints = tuple(valp)

            yield constraint, hints

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
                'max_util_drop={}'.format(repr(self.max_util_drop)),
                'with_opt={}'.format(repr(self.with_opt))]))

    def _key_attrs(self):
        ''' Used for comparison. '''
        return (self.seg, self.network, self.batch_size, self.resource,
                self.max_util_drop, self.with_opt)

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
        Memory sources and neighbor sources must be mutual exclusive, in order
        to correctly set the src data regions; memory destinations and neighbor
        destinations can co-exist.

        Local dependencies are omitted, as by default each layer has its
        immediately previous layer as local source and immediately next layer
        as local destination.

        Construct an ifmap forwarding dict for shared memory source data. It
        maps previous layer name tuples, to a list of scheduling indices of all
        layers in this segment that share these exact previous layers. The
        first in the list is responsible to fetch the previous layer data and
        to forward them to others. We allow shared memory source data between
        two layers only when both layers have memory dependency only (so their
        temporal indices must be 0), and their previous layers are exactly the
        same.

        Construct an ofmap forwarding dict for multiple destinations of both
        on-chip and off-chip. It maps the scheduling index of a layer in this
        segment that has both memory and neighbor/local destinations (so needs
        to store its ofmaps back to memory), to a list of scheduling indices of
        all layers in this segment that accepts its ofmaps as ifmaps. Neighbor
        dependencies are only between the last temporal one and the first
        temporal ones; local dependencies are only between adjacent temporal
        ones.
        '''

        self.src_dict = [[None for _ in ltpl] for ltpl in self.seg]
        self.dst_dict = [[None for _ in ltpl] for ltpl in self.seg]

        self.ifm_fwd_dict = {}
        self.ofm_fwd_dict = {}

        # Mapping from layer to spatial/temporal indices in the segment.
        layer2idx = {l: PipelineSegment.SchedIndex(sp_idx, tm_idx)
                     for sp_idx, ltpl in enumerate(self.seg)
                     for tm_idx, l in enumerate(ltpl)}

        # Mapping from previous layer tuple to layer.
        prevs2layer = {}

        for sp_idx, ltpl in enumerate(self.seg):

            single_nbr_src = None

            for tm_idx, l in enumerate(ltpl):

                assert layer2idx[l] == (sp_idx, tm_idx)

                # Sources.
                src = tuple()

                prevs = self.network.prevs(l)
                assert all(p not in layer2idx or layer2idx[p] < layer2idx[l]
                           for p in prevs)
                mem_src = [p for p in prevs if p not in layer2idx]
                lcl_src = [p for p in prevs if p not in mem_src
                           and layer2idx[p].sp_idx == sp_idx]
                nbr_src = [p for p in prevs if p not in mem_src + lcl_src]

                # Ensure single local source to be the immediately previous.
                # Check at the destination so here are assertions.
                if not lcl_src:
                    assert tm_idx == 0
                else:
                    assert len(lcl_src) == 1 \
                            and layer2idx[lcl_src[0]].tm_idx == tm_idx - 1

                # Mutual exclusive.
                if mem_src and nbr_src:
                    # We now allow each spatial scheduling (vertex) to have
                    # both memory source and neighbor source when generating
                    # segments. But each single layer cannot have both;
                    # otherwise there would be multiple source data regions.
                    return False

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
                    if single_nbr_src is not None:
                        return False
                    single_nbr_src = prev_idx
                    src += (prev_idx,)

                # Shared memory source.
                if mem_src and not lcl_src:
                    assert not nbr_src
                    assert tm_idx == 0
                    if prevs in prevs2layer:
                        fet_idx = layer2idx[prevs2layer[prevs]]
                        self.ifm_fwd_dict.setdefault(prevs, [fet_idx]).append(
                            layer2idx[l])
                    else:
                        prevs2layer[prevs] = l

                # Destinations.
                dst = tuple()

                nexts = self.network.nexts(l)
                assert all(n not in layer2idx or layer2idx[n] > layer2idx[l]
                           for n in nexts)
                mem_dst = [n for n in nexts if n not in layer2idx]
                lcl_dst = [n for n in nexts if n not in mem_dst
                           and layer2idx[n].sp_idx == sp_idx]
                nbr_dst = [n for n in nexts if n not in mem_dst + lcl_dst]

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
                # Now they can co-exist.
                # assert not mem_dst or not nbr_dst
                if mem_dst and nbr_dst:
                    assert tm_idx == len(ltpl) - 1
                    self.ofm_fwd_dict[layer2idx[l]] = [layer2idx[n]
                                                       for n in nbr_dst]
                if mem_dst and lcl_dst:
                    assert not nbr_dst
                    self.ofm_fwd_dict[layer2idx[l]] = [layer2idx[lcl_dst[0]]]

                if mem_dst:
                    # Memory destination.
                    dst += (None,)
                if nbr_dst:
                    # Neighbor destinations.
                    # This layer is the last temporal scheduled.
                    assert tm_idx == len(ltpl) - 1
                    dst += tuple(layer2idx[n] for n in nbr_dst)

                # Basic pipelining requires a linear structure (on-chip).
                if not self.with_opt:
                    if len(nbr_src) + len(lcl_src) > 1 \
                            or len(nbr_dst) + len(lcl_dst) > 1 \
                            or ((sp_idx, tm_idx) != (0, 0)
                                    and not nbr_src and not lcl_src):
                        return False

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

        no_time_mux = len(self.network) == sum(len(ltpl) for ltpl in self.seg)
        # All layers that have model filters must be spatially scheduled.
        if no_time_mux:
            for ltpl in self.seg:
                if len([l for l in ltpl
                        if isinstance(self.network[l], ConvLayer)]) > 1:
                    no_time_mux = False
                    break

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
                    src_data_region = self.resource.src_data_region
                    for sh_idx_list in self.ifm_fwd_dict.values():
                        # Find shared memory source to use forwarding.
                        if (sp_idx, tm_idx) in sh_idx_list[1:]:
                            src_data_region = subregions[sh_idx_list[0].sp_idx]
                            break
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
                    # assert dst == (None,)
                    # Now we can have both memory and neighbor destinations. If
                    # they co-exist, we need to store them locally and also
                    # store back to memory. In this case the dst data region is
                    # set to memory.
                    dst_data_region = self.resource.dst_data_region
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
                rtpl += (self.resource._replace(
                    proc_region=proc_region,
                    src_data_region=src_data_region,
                    dst_data_region=dst_data_region,
                    no_time_mux=no_time_mux),)

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
            nodes = [max(1, int(round(nr / cf))) * cf for nr in nodes_raw]
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

    def _init_sym_cstrs(self):
        '''
        Initialize the symbolic scheduling constraints for the layers in the
        segment, by constructing a nested lists of dicts `cstr_symargs` whose
        values can be symbolic expressions for the keyword arguments of layers
        in the segment, and a dict `cstr_symvals` mapping each symbol to its
        possible numerical values.

        Rules for constraints.

        - Top BAT loop factor.

        With a single layer, there is no constraint on the top BAT loop factor.
        Otherwise all layers must share the same factor, namely `topbat_shr`.

        - Fmap forwarding and fully buffering.

        Only CONV layers require to fully buffer fmaps. Local-region layers
        process data in a streaming manner.

        Each CONV layer, and all local-region layers immediately following it
        within the same spatial scheduling, are made into a group G.

        (initial) if G is both the first spatial and the first temporal
        scheduling with a CONV layer, it can choose whether to fully buffer
        ofmaps or not. This is a configuration to explore, namely `fbofm_init`.
        We decide its value by choosing the one that gives the fewer fully
        buffered inter-spatial pairs on the critical forwarding path, and the
        smaller maximum fully buffered data size.

        (within-group) within G, the CONV layer, and all local-region layers,
        should use the same top OFM factors (IFM factors are automatically
        determined by OFM factors in local-region layers), unless CONV ofmaps
        need to be fully buffered, in which case, the CONV layer and the last
        layer in G fully buffer ofmaps (top OFM factor is 1), and other layers
        still use the same top OFM factors but can be different than 1.

        (inter-temporal) if G has a source from G' in the same spatial
        scheduling (which must be immediately before G), G should fully buffer
        ifmaps, and G' should fully buffer ofmaps.

        (inter-spatial) if G has a source from G' in another spatial scheduling
        (where the source must be the last temporal scheduling in G' and that
        spatial scheduling),
        (a) if G' already fully buffers ofmaps, make G fully buffer ifmaps.
        (b) otherwise, make G fully buffer ofmaps (do not require G' to fully
            buffer ifmaps; leave it to other rules, e.g. inter-temporal, to
            decide); forward data between G' and G, by matching their top O/IFM
            factors (biasing this case for smaller pipeline filling delay).
        Notice the destination can be: (1) the leading CONV layer, whose top
        IFM factor is constrained; (2) a local-region layer, where we constrain
        the top OFM factors of this group (except otherwise constrained by
        fully buffering ofmaps).
        '''
        # pylint: disable=too-many-branches

        # Symbolic variables mapping to numerical values.
        symvals = dict()

        # Top BAT loop factor.
        topbat = symbols('topbat_shr', integer=True)
        symvals[topbat] = [t for t, _ in util.factorize(self.batch_size, 2)]

        # Whether the initial CONV layer fully buffers ofmaps.
        fbofm_init = symbols('fbofm_init')
        symvals[fbofm_init] = [False, True]

        def _layer_topofm_vals(layer_name):
            layer = self.network[layer_name]
            # We require that the total ofmap size takes at least 5% of the
            # gbuf capacity of a single node, to avoid too fine blocking.
            tmax = layer.total_ofmap_size(self.batch_size) \
                    / (0.05 * self.resource.size_gbuf)
            vals = [t for t, _ in util.factorize(layer.nofm, 2)
                    if t <= tmax or t == 1]
            assert vals
            return vals

        def _layer_topifm_vals(layer_name):
            layer = self.network[layer_name]
            # We require that the total ifmap size takes at least 5% of the
            # gbuf capacity of a single node, to avoid too fine blocking.
            tmax = layer.total_ifmap_size(self.batch_size) \
                    / (0.05 * self.resource.size_gbuf)
            vals = [t for t, _ in util.factorize(layer.nifm, 2)
                    if t <= tmax or t == 1]
            assert vals
            return vals

        # Layer constraint kwargs.
        symargs = [[{'topbat': topbat} for _ in ltpl] for ltpl in self.seg]

        # Candidates for critical forwarding path between spatial scheduling.
        sp_crit_path_cands = set()
        sp_crit_path_cands.add((0,))  # init with the first spatial.

        # The last CONV layer index.
        last_conv = PipelineSegment.SchedIndex(-1, 0)

        # Whether the current group needs to fully buffer ofmap. Delayed apply
        # to the last layer in the group.
        curr_fbofm = False

        for sp_idx, ltpl in enumerate(self.seg):

            # Initial topofm, in case of a non-CONV starting layer.
            curr_topofm = symbols('topofm_{}_s'.format(sp_idx), integer=True)
            symvals[curr_topofm] = _layer_topofm_vals(ltpl[0])

            for tm_idx, l in enumerate(ltpl):

                layer = self.network[l]
                curr_sa = symargs[sp_idx][tm_idx]

                # Neighbor source dependency.
                nsrc_sa = None
                src_deps = self.src_dict[sp_idx][tm_idx]
                if any(s is not None for s in src_deps):
                    assert len(src_deps) == 1
                    nbr_src = src_deps[0]
                    assert nbr_src.sp_idx < sp_idx
                    nsrc_sa = symargs[nbr_src.sp_idx][nbr_src.tm_idx]
                    assert nsrc_sa  # not empty, used to test nbr src exists.
                    # Set critical path candidates.
                    new_cands = set()
                    for cand in sp_crit_path_cands:
                        if cand[-1] == nbr_src.sp_idx:
                            new_cands.add(cand + (sp_idx,))
                    sp_crit_path_cands |= new_cands

                if isinstance(layer, ConvLayer):
                    # Conv layer.

                    # The last group may require to fully buffer ofmaps.
                    # Delayed apply to the immediate previous layer.
                    if curr_fbofm is not False:
                        assert last_conv >= (0, 0)
                        if last_conv.sp_idx == sp_idx:
                            assert tm_idx > 0
                            lsrc_sa = symargs[sp_idx][tm_idx - 1]
                        else:
                            lsrc_sa = symargs[last_conv.sp_idx][-1]
                        lsrc_sa['fbofm'] = curr_fbofm
                    # Reset.
                    curr_fbofm = False

                    # New topofm for a new group.
                    curr_topofm = symbols('topofm_{}_{}'.format(sp_idx, tm_idx),
                                          integer=True)
                    symvals[curr_topofm] = _layer_topofm_vals(l)

                    # Set topofm.
                    curr_sa['topofm'] = curr_topofm

                    if sp_idx == last_conv.sp_idx:
                        # Rule inter-temporal.
                        assert tm_idx > 0
                        # Make this group fully buffer ifmaps.
                        curr_sa['fbifm'] = True
                        # Make the last group fully buffer ofmaps.
                        last_sa = symargs[sp_idx][last_conv.tm_idx]
                        lsrc_sa = symargs[sp_idx][tm_idx - 1]
                        last_sa['fbofm'] = True
                        lsrc_sa['fbofm'] = True

                    elif nsrc_sa:
                        # Rule inter-spatial.
                        # We only look at this rule when inter-temporal rule
                        # does not apply and the ifmaps of this group are not
                        # yet required to fully buffer.
                        if not self.with_opt:
                            # Basic pipelining requires fully buffering all
                            # pairs of neighbor src/dst.
                            nsrc_sa['fbofm'] = True
                        nsrc_fbofm = nsrc_sa.get('fbofm', False)
                        # (a): if the source already fully buffers ofmaps.
                        # Make this group fully buffer ifmaps.
                        curr_sa['fbifm'] = symeq(nsrc_fbofm, True)
                        # (b)-(1): otherwise.
                        # Make this group fully buffer ofmaps.
                        curr_sa['fbofm'] = symeq(nsrc_fbofm, False)
                        curr_fbofm = symeq(nsrc_fbofm, False)  # delayed apply.
                        # Match top OFM/IFM factors.
                        curr_sa['topifm'] = sympiecewise(
                            (nsrc_sa['topofm'], symeq(nsrc_fbofm, False)),
                            (curr_sa.get('topifm', 0), True))

                    elif last_conv < (0, 0):
                        # The first CONV layer.
                        # Rule initial.
                        curr_sa['fbofm'] = fbofm_init
                        curr_fbofm = fbofm_init

                    last_conv = PipelineSegment.SchedIndex(sp_idx, tm_idx)

                else:
                    # Non-Conv layer.

                    if nsrc_sa:
                        # Rule inter-spatial, (b)-(2).
                        nsrc_fbofm = nsrc_sa.get('fbofm', False)
                        curr_topofm = sympiecewise(
                            (nsrc_sa['topofm'], symeq(nsrc_fbofm, False)),
                            (curr_topofm, True))
                        # Also backtrace this group.
                        for bt_idx in range(last_conv.tm_idx, tm_idx):
                            symargs[sp_idx][bt_idx]['topofm'] = curr_topofm

                    # Rule within-group.
                    curr_sa['topofm'] = curr_topofm

                # If this layer has no on-chip destinations, cancel the
                # requirement to fully buffer ofmaps.
                if all(d is None for d in self.dst_dict[sp_idx][tm_idx]) \
                        and tm_idx == len(ltpl) - 1:
                    curr_sa.pop('fbofm', False)

        # Simplify.
        self._simplify_symargs(symargs, symvals)

        # Get critical forwarding path between spatial scheduling.
        # The critical path has the longest forwarding chain.
        sp_crit_path = max(sp_crit_path_cands, key=len)

        # Check maximum fully-buffering size, and decide fbofm_init.
        opt_val = None
        opt_key = (float('inf'),) * 2  # (num of fb pairs, max fb size)
        num_sp_fbs = 0
        for val in symvals.get(fbofm_init, [False]):
            subs_symargs = self._subs_symargs(symargs, fbofm_init, val)
            maxsz = 0
            numfb = 0
            for sp_idx, (ltpl, atpl) in enumerate(zip(self.seg, subs_symargs)):
                ms = max(itertools.chain(
                    ((self.network[l].total_ofmap_size() if a.get('fbofm')
                      else 0)
                     + (self.network[l].total_ifmap_size() if a.get('fbifm')
                        else 0)
                     for l, a in zip(ltpl, atpl)),
                    [0]))  # safe max with default.
                if ms > self.alloc[sp_idx][0].proc_region.dim.size() \
                        * self.alloc[sp_idx][0].size_gbuf:
                    break
                maxsz = max(maxsz, ms)
                if sp_idx in sp_crit_path and atpl[-1].get('fbofm', False):
                    numfb += 1
            else:
                key = (numfb, maxsz)
                if key < opt_key:
                    opt_val, opt_key = val, key
                    num_sp_fbs = numfb
        if opt_val is None:
            return False
        # Use the optimal value.
        symvals[fbofm_init] = [opt_val]
        self._simplify_symargs(symargs, symvals)

        # Shared memory source must have the same topifm.
        for sh_idx_list in self.ifm_fwd_dict.values():
            assert len(sh_idx_list) > 1
            fet_sp_idx = sh_idx_list[0].sp_idx
            sh_symarg_list = [symargs[idx.sp_idx][0] for idx in sh_idx_list]

            # Must have no constraint on ifmaps access from memory.
            assert all(not sa.get('fbifm', False) and not sa.get('topifm', 0)
                       for sa in sh_symarg_list)

            # Cannot constrain both topifm and topofm.
            if any(sa.get('fbofm', False) or sa.get('topofm', 0)
                   for sa in sh_symarg_list):
                sh_kwargs = {'fbifm': True}
            else:
                topifm = symbols('topifm_{}'.format(fet_sp_idx), integer=True)
                symvals[topifm] = _layer_topifm_vals(self.seg[fet_sp_idx][0])
                sh_kwargs = {'topifm': topifm}

            # Set constraints.
            for sa in sh_symarg_list:
                sa.update(sh_kwargs)

        # Simplify.
        self._simplify_symargs(symargs, symvals)

        # Turn constraints into lazily updated rules.
        self._lazify_topofm_symargs(symargs, symvals)
        # Cannot simplify any more as update_dict is not sympifi-able.

        # Sort symbol dict.
        symvals = OrderedDict(sorted(((s, symvals[s]) for s in symvals),
                                     key=lambda item: str(item[0])))

        if not symvals:
            # Must add a dummy symbol so iterative substitution can happen.
            symvals[symbols('_dummy')] = [None]

        self.cstr_symargs = symargs
        self.cstr_symvals = symvals
        self.cstr_num_sp_fbs = num_sp_fbs
        try:
            self.cstr_topbat_idx = list(symvals.keys()).index(topbat)
        except ValueError:
            self.cstr_topbat_idx = None

        return True

    @staticmethod
    def _simplify_symargs_one_pass(symargs, symvals):
        '''
        Simplify symargs and symvals in-place:
        - If fbi/ofm is False, then remove it.
        - If fbi/ofm is True, then remove topi/ofm.
        - If a symbol can take only one value, then substitute it.
        - If a symbol only occurs once, then remove its constraint.

        Return whether the symargs and symvals are already simplified.
        '''
        for a in itertools.chain.from_iterable(symargs):
            is_fbifm = a.get('fbifm')
            is_fbofm = a.get('fbofm')
            # pylint: disable=singleton-comparison
            # lhs may be symbolic, see
            # docs.sympy.org/latest/modules/logic.html#sympy.logic.boolalg.BooleanTrue
            if is_fbifm == True:
                a.pop('topifm', 0)
            if is_fbifm == False:
                a.pop('fbifm', False)
            if is_fbofm == True:
                a.pop('topofm', 0)
            if is_fbofm == False:
                a.pop('fbofm', False)

        subs_dict = {}

        # Possible values for symbols.
        subs_dict.update(
            (s, symvals[s][0]) for s in symvals if len(symvals[s]) == 1)

        # Count the occurrence of symbols in all args (values).
        symcnts = Counter(
            s for a in itertools.chain.from_iterable(symargs)
            for val in a.values() for s in symtuple(val).free_symbols)
        assert set(symcnts.keys()).issubset(symvals.keys())
        subs_dict.update((s, None)
                         for s in set(symvals.keys()) - set(symcnts.keys()))
        subs_dict.update((s, 0 if str(s).startswith('top') else False)
                         for s in symcnts if symcnts[s] <= 1)

        # Substitute symbols and remove from symbol dict.
        for a in itertools.chain.from_iterable(symargs):
            for k in a:
                a[k] = symtuple(a[k]).subs(subs_dict)[0]
        for s in subs_dict:
            del symvals[s]

        return not subs_dict

    def _simplify_symargs(self, symargs, symvals):
        ''' Simplify symargs and symvals in-place iteratively. '''
        while not self._simplify_symargs_one_pass(symargs, symvals):
            pass
        used_syms = symtuple(
            *[symtuple(*a.values())
              for a in itertools.chain.from_iterable(symargs)]).free_symbols
        assert set(used_syms) == set(symvals.keys())
        assert all(val for val in symvals.values())

    @staticmethod
    def _subs_symargs(symargs, *subs_args):
        '''
        Substitute symbols. The additional arguments are passed to subs().

        Return a new substituted copy without modifying the original one.
        '''
        # sympify=False is necessary because there may be str in the values.
        return [[dict((k, symtuple(a[k], sympify=False).subs(*subs_args)[0])
                      for k in a) for a in atpl] for atpl in symargs]

    class TopOfmUpdateLambda(symbasic):
        ''' A sympifi-able lambda function to lazily update topofm. '''
        # pylint: disable=no-init
        def __new__(cls, *args):
            return super(PipelineSegment.TopOfmUpdateLambda, cls).__new__(cls)
        def __call__(self, arg_s, arg_r):
            setattr(arg_s, 'topofm', arg_r.scheme['to'][0])

    def _lazify_topofm_symargs(self, symargs, symvals):
        '''
        Turn qualified topofm constraints into lazily updated rules.

        If a symbol is only used as the topofm constraint by a single CONV
        layer and some local-region layers, we can turn it into a lazily update
        rule.
        '''
        sym2conv = {}  # symbol --> the only CONV layer using it.
        sym2lrs = {}   # symbol --> list of local-region layer using it.
        unqual_syms = set()  # symbols used by two or more CONV layers.
        for l, a in zip(itertools.chain.from_iterable(self.seg),
                        itertools.chain.from_iterable(symargs)):
            layer = self.network[l]
            if isinstance(layer, ConvLayer):
                topofm = a.get('topofm', 0)
                topifm = a.get('topifm', 0)
                for s in symtuple(topofm, topifm).free_symbols:
                    if s not in unqual_syms:
                        if s in sym2conv:
                            # If a symbol is used in two CONV layers, it cannot
                            # be lazily updated.
                            del sym2conv[s]
                            sym2lrs.pop(s, [])
                            unqual_syms.add(s)
                        elif topofm == s:
                            assert s not in sym2lrs
                            sym2conv[s] = l
            else:
                topofm = a.get('topofm', 0)
                if topofm in sym2conv:
                    sym2lrs.setdefault(topofm, []).append(l)
        assert 0 not in sym2conv and 0 not in sym2lrs

        syms = sym2conv.keys()  # symbols to be lazily updated.
        lr2conv = {}  # local-region layer to the CONV layer constraining it.
        for s in syms:
            for lr in sym2lrs.get(s, []):
                lr2conv[lr] = sym2conv[s]
        lconvs = set(lr2conv.values())  # CONV layer whose topofm to be removed.

        for l, a in zip(itertools.chain.from_iterable(self.seg),
                        itertools.chain.from_iterable(symargs)):
            if l in lconvs:
                # Remove CONV topofm.
                assert sym2conv[a['topofm']] == l
                del a['topofm']
            elif l in lr2conv:
                # Link local-region layer to the CONV layer.
                lconv = lr2conv[l]
                assert sym2conv[a['topofm']] == lconv
                del a['topofm']
                a['update_dict'] = {
                    lconv: PipelineSegment.TopOfmUpdateLambda()}

        for s in syms:
            del symvals[s]

