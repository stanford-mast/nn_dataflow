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

from . import Util
from .Layer import ConvLayer
from .Network import Network
from .Resource import Resource

class InterLayerPipeline(object):
    '''
    Inter-layer pipeline.
    '''

    def __init__(self, network, resource):
        if not isinstance(network, Network):
            raise TypeError('InterLayerPipeline: network must be '
                            'a Network instance.')
        if not isinstance(resource, Resource):
            raise TypeError('InterLayerPipeline: resource must be '
                            'a Resource instance.')

        self.network = network
        self.resource = resource

        self._calc_sched_dag()

        # Vertices starting from which we have generated the segments.
        self.seg_vertex_done = set()

    def ordered_layer_list(self):
        '''
        Get a list of the layers in their topological order in the scheduling
        DAG.
        '''
        return list(sum(self.dag_vertex_list, tuple()))

    def gen_segment_allocation(self, options, max_util_drop=0.05):
        '''
        Generate all inter-layer pipelining segments and their resource
        allocations.

        Return a segment layer tuple, and a resource allocation tuple. The two
        tuples contains sub-tuples, where different sub-tuples are spatially
        scheduled, and different layers in a sub-tuple is temporally scheduled.
        '''

        if not (options.partition_interlayer or options.hw_gbuf_save_writeback):
            # No inter-layer pipelining, each vertex sequentially occupies the
            # whole resource.
            for layer in self.network:
                yield ((layer,),), ((self.resource,),)
            return

        for segment in self._gen_segment():

            if options.partition_interlayer:

                segalloc = self._allocate_segment(segment,
                                                  max_util_drop=max_util_drop)
                if segalloc:
                    yield segalloc

            if options.hw_gbuf_save_writeback:

                segalloc = self._allocate_segment(segment, temporal=True)
                if segalloc:
                    yield segalloc

    def _gen_segment(self, vertex_idx=0, done=None):
        '''
        Generate segments starting from vertex `vertex_idx`. Yield a tuple of
        the vertices in the segment.

        `done` is a set of vertices which have already been scheduled and the
        output is already in memory.

        Rules:

        1. If a vertex does not share any dependencies with the current
        segment, i.e., none of its previous vertices is in the current segment
        or among the previous vertices of the current segment, we do not add it
        to the segment, because there is no benefit to co-locate them.

        2. If a vertex has multiple previous vertices, none of them
        can be in the same segment as this vertex, because the output data
        availability timing of the previous vertices may not match.

        3. If a vertex has multiple next vertices, all or none of them can be
        in the same segment as this vertex, because only including a subset of
        the next vertices cannot eliminate the data write-back to memory.
        '''

        segment = tuple()

        if not done:
            done = set()

        if self.dag_input_vertex not in done:
            # Input layer is always in memory.
            done.add(self.dag_input_vertex)

        # The frontier is the vertex to be considered to be added to the
        # current segment.
        for frontier in range(vertex_idx, len(self.dag_vertex_list)):

            # Check whether the frontier can be added to the current segment.

            frontier_prevs = self.dag_prev_dict[frontier]

            # Whether the frontier share dependencies with the current segment,
            # if the segment is not empty.
            share_deps = not segment \
                    or not frontier_prevs.isdisjoint(
                        set.union(set(segment),
                                  *[self.dag_prev_dict[i] for i in segment]))

            # Whether some of the multiple previous vertices are in the current
            # segment.
            coupled_prevs = len(frontier_prevs) > 1 \
                    and not frontier_prevs.isdisjoint(segment)

            if not share_deps or coupled_prevs:
                # Not sharing any dependencies (rule 1), or previous vertices
                # overlap with the current segment (rule 2).

                # Make sure the current segment is not empty.
                assert segment
                # Not extend the segment any more. Note that the current
                # segment has already been yielded, as well as the recursion,
                # in the last iteration.
                break

            # Extend the segment.
            segment += (frontier,)

            # Check whether the segment is valid.

            for idx in segment:
                nexts = self.dag_next_dict[idx]

                # The next vertices should either all or none in the segment
                # (rule 3).
                if not (nexts.isdisjoint(segment) or nexts.issubset(segment)):
                    # The segment is invalid. Need to add more vertices.
                    assert min(nexts.difference(segment)) > frontier
                    break
            else:
                # The segment is valid.
                yield segment

                # Skip if have done.
                if frontier + 1 in self.seg_vertex_done:
                    continue

                # Recursion.
                for tpl in self._gen_segment(frontier + 1,
                                             done.union(segment)):
                    yield tpl

        assert vertex_idx not in self.seg_vertex_done
        self.seg_vertex_done.add(vertex_idx)

    def _allocate_segment(self, segment, temporal=False, max_util_drop=0.05):
        '''
        Allocate resource to the vertices in the given `segment`.

        Return a segment layer tuple, and a resource allocation tuple. The two
        tuples contains sub-tuples, where different sub-tuples are spatially
        scheduled, and different layers in a sub-tuple is temporally scheduled.
        Return None if allocation failed.

        If `temporal` is True, the resource is allocated temporally to the
        vertices in the given `segment`. Each layer in the segment will
        sequentially use all the resources.

        `max_util_drop` specifies the maximum utilization drop due to mismatch
        throughput between vertices.
        '''

        assert segment

        # The segment layers.
        layer_list = []

        for vidx in segment:
            layer_list.append(self.dag_vertex_list[vidx])

        if temporal:
            # Reduce the spatial dimension.
            layer_list = [sum(layer_list, tuple())]

            # Check. The spatial allocation won't have multiple previous layers
            # feed a single layer, but could have a single layer feed multiple
            # next layers. The latter case is not valid for temporal
            # allocation, as there is nowhere to keep the output.
            for layer in layer_list[0]:
                if sum(nl in layer_list[0]
                       for nl in self.network.next_layers(layer)) > 1:
                    return None

        # Spatial allocation.
        proc_region = self.resource.proc_region
        dim_nodes = proc_region.dim
        total_nodes = dim_nodes.size()

        ops = [self.dag_vertex_ops[vidx] for vidx in segment]
        if temporal:
            ops = [sum(ops)]

        # Enforce a common factor among the numbers of nodes allocated to all
        # vertices in the segment. Such common factor is likely to be the
        # common height of the vertex node regions.
        common_factor_list = [cf for cf, _ in Util.factorize(dim_nodes.h, 2)]

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
            util = sum(ops) / time / sum(nodes)
            assert util < 1 + 1e-6

            if util >= 1 - max_util_drop:
                # Found
                break

        else:
            # Not found.
            return None

        # Allocate in the processing region according to the number of nodes.
        subregions = proc_region.allocate(nodes)
        assert subregions

        if temporal:
            assert len(subregions) == 1
            assert subregions[0] == proc_region

        # The resource allocation.
        resource_list = []

        layer2sp = dict((l, sp_idx)
                        for sp_idx, ltpl in enumerate(layer_list)
                        for l in ltpl)

        for ltpl, sr in zip(layer_list, subregions):

            rtpl = tuple()

            for tm_idx, layer in enumerate(ltpl):

                # Data source.
                prev_layers, _ = self.network.prev_layers(layer)

                local_prev_layers = [pl for pl in prev_layers if pl in ltpl]
                if local_prev_layers:
                    assert len(local_prev_layers) == 1
                    assert local_prev_layers[0] == ltpl[tm_idx - 1]

                # Non-local data source.
                # We ignore the local data sources.
                nonlocal_prev_layers = [pl for pl in prev_layers
                                        if pl not in ltpl]

                if not nonlocal_prev_layers:
                    # Data source is local.
                    src_data_region = sr
                elif any(pl in layer2sp for pl in nonlocal_prev_layers):
                    # Data source is from the same segment.
                    assert len(nonlocal_prev_layers) == 1
                    prev_sp_idx = layer2sp[nonlocal_prev_layers[0]]
                    src_data_region = subregions[prev_sp_idx]
                else:
                    # Data source is from memory.
                    src_data_region = self.resource.src_data_region()

                # Data destination.
                next_layers = self.network.next_layers(layer)

                local_next_layers = [nl for nl in next_layers if nl in ltpl]
                if local_next_layers:
                    assert len(local_next_layers) == 1
                    assert local_next_layers[0] == ltpl[tm_idx + 1]

                # Non-local data destination.
                # We ignore the local data destinations.
                nonlocal_next_layers = [nl for nl in next_layers
                                        if nl not in ltpl]

                if not nonlocal_next_layers:
                    # Data destination is local.
                    dst_data_region = sr
                elif any(nl in layer2sp for nl in nonlocal_next_layers):
                    # Data destination is to the same segment.
                    assert len(nonlocal_next_layers) == 1
                    # Put data in local. The next layer will fetch.
                    dst_data_region = sr
                else:
                    # Data destination is to memory.
                    dst_data_region = self.resource.dst_data_region()

                rtpl += (Resource(proc_region=sr,
                                  data_regions=(src_data_region,
                                                dst_data_region),
                                  dim_array=self.resource.dim_array,
                                  size_gbuf=self.resource.size_gbuf,
                                  size_regf=self.resource.size_regf),)

            assert len(ltpl) == len(rtpl)
            resource_list.append(rtpl)

        assert len(layer_list) == len(resource_list)
        return tuple(layer_list), tuple(resource_list)

    def _calc_sched_dag(self):
        '''
        Build the scheduling DAG of the network. We merge layers with no
        filters into their last previous layer, so a DAG vertex can contain one
        or more layers.

        We order and index the DAG vertices in their depth-first topological
        order. This will also be the order to schedule the layers.

        Also establish two dicts for the previous and next vertices of each DAG
        vertex.

        Also record the number of operations of each DAG vertex.

        In summary, the attributes initialized include: `dag_input_vertex`,
        `dag_vertex_list`, `dag_vertex_dict`, `dag_prev_dict`, `dag_next_dict`,
        `dag_vertex_ops`.
        '''

        # Vertex of the input layer.
        self.dag_input_vertex = -1

        # The DAG vertex set. Each vertex is a merged layer tuples, represented
        # by their layer names. Use a list type to make modification easier.
        dag_vertex_set = []

        for layer_name in self.network:
            layer = self.network[layer_name]

            if isinstance(layer, ConvLayer):
                dag_vertex_set.append((layer_name,))

            else:
                prev_layers, _ = self.network.prev_layers(layer_name)
                assert prev_layers
                last_prev = prev_layers[-1]

                if not last_prev:
                    # Only has the input layer as the previous layer. Nothing
                    # to merge.
                    dag_vertex_set.append((layer_name,))

                else:
                    # Find and merge to the last previous layer vertex.
                    found = False
                    for idx in reversed(range(len(dag_vertex_set))):
                        if last_prev in dag_vertex_set[idx]:
                            dag_vertex_set[idx] += (layer_name,)
                            found = True
                            break
                    assert found

        assert sum(len(v) for v in dag_vertex_set) == len(self.network)

        # The DAG vertex list in the topological order.
        self.dag_vertex_list = self._topological_order(dag_vertex_set)

        # Make a directory from layer name to DAG vertex index.
        self.dag_vertex_dict = {}

        for vidx, v in enumerate(self.dag_vertex_list):
            for layer_name in v:
                assert layer_name not in self.dag_vertex_dict
                self.dag_vertex_dict[layer_name] = vidx

        # Add the input layer.
        self.dag_vertex_dict[self.dag_input_vertex] = \
                self.network.INPUT_LAYER_KEY

        # The previous and next relationship of the DAG vertices.
        self.dag_prev_dict = dict((vidx, set()) for vidx
                                  in range(len(self.dag_vertex_list)))
        self.dag_next_dict = dict((vidx, set()) for vidx
                                  in range(len(self.dag_vertex_list)))

        for layer_name in self.network:
            vidx = self.dag_vertex_dict[layer_name]

            # Previous layers.
            prev_layers, _ = self.network.prev_layers(layer_name)
            for pl in prev_layers:
                pvidx = self.dag_vertex_dict[pl] if pl \
                        else self.dag_input_vertex
                if pvidx != vidx:
                    self.dag_prev_dict[vidx].add(pvidx)

            # Next layers.
            next_layers = self.network.next_layers(layer_name)
            for nl in next_layers:
                if not nl:
                    continue
                nvidx = self.dag_vertex_dict[nl]
                if nvidx != vidx:
                    self.dag_next_dict[vidx].add(nvidx)

        # Add next layers of the input layer.
        self.dag_next_dict[self.dag_input_vertex] = set()
        for vidx in self.dag_prev_dict:
            if self.dag_input_vertex in self.dag_prev_dict[vidx]:
                self.dag_next_dict[self.dag_input_vertex].add(vidx)

        # Number of ops of each vertex.
        self.dag_vertex_ops = []
        for v in self.dag_vertex_list:
            ops = sum(self.network[l].total_ops() for l in v)
            self.dag_vertex_ops.append(ops)

    def _topological_order(self, dag_vertex_set):
        '''
        Order the DAG vertices in topological order using DFS.

        Specifically, The backtrace order of the depth-first search is the
        inverse of the topological order. See
        https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
        '''

        # The visited layers in the DFS order.
        visited = []
        # The unseen pending layers.
        unseen = set(dag_vertex_set)
        # The layers that have been seen, but not visited due to unvisited
        # previous layers.
        seen = set()

        def _dfs(vertex):
            assert vertex not in seen
            if vertex in visited:
                return

            unseen.discard(vertex)
            seen.add(vertex)

            next_layers = []
            for l in vertex:
                for nl in self.network.next_layers(l):
                    if nl and nl not in vertex and nl not in next_layers:
                        next_layers.append(nl)

            # Visit next layers in the reversed order, so the reversed visit
            # order has the original order.
            next_vertices = []
            for nl in reversed(next_layers):
                for nv in unseen:
                    if nl in nv:
                        next_vertices.append(nv)

            for nv in next_vertices:
                _dfs(nv)

            visited.append(vertex)
            seen.remove(vertex)

        # Start from the first layers.
        start_vertices = []
        for l in reversed(self.network.first_layers()):
            for v in unseen:
                if l in v:
                    start_vertices.append(v)
        for v in start_vertices:
            _dfs(v)
        assert not unseen
        assert not seen

        return list(reversed(visited))

