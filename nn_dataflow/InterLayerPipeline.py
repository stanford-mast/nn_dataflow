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

    def _gen_segment(self, seg_idx, vertex_idx, done, no_repeating=False):
        '''
        Generate segments starting from segment index `seg_idx`, with starting
        vertex `vertex_idx`. Yield the segment index, and a tuple of the
        vertices in the segment.

        Segment index is a sequence number that back-traces when reaching the
        end of the network. If it back-traces to N, the previously yielded
        segments with index smaller than N will be reused.

        `done` is a set of vertices which have already been scheduled and the
        output is already in memory.

        If `no_repreating` is True, we do not re-generate the same segments
        with the same `vertex_idx` for different prefixes. For example, if we
        have generated the segments starting from vertex 2 for prefix (0), (1),
        then we do not generate these segments for prefix (0, 1). To make it
        work, the upper level should cache the results.

        Rules:

        1. If a vertex does not share any dependencies with the current
        segment, i.e., none of its previous vertices is in the current segment
        or among the previous vertices of the current segment, we do not add it
        to the segment, because there is no benefit to co-locate them.

        2. If a vertex has multiple previous vertices, no more than one of them
        can be in the same segment as this vertex, because the output data
        availability timing of the previous vertices may not match.

        3. If a vertex has multiple next vertices, all or none of them can be
        in the same segment as this vertex, because only including a subset of
        the next vertices cannot eliminate the data write-back to memory.
        '''

        segment = tuple()

        if self.dag_input_vertex not in done:
            # Input layer is always in memory.
            done.add(self.dag_input_vertex)

        # The frontier is the vertex to be considered to be added to the
        # current segment.
        for frontier in range(vertex_idx, len(self.dag_vertex_list)):

            # Check whether the frontier can be added to the current segment.

            # Whether the frontier share dependencies with the current segment,
            # if the segment is not empty.
            share_deps = not segment \
                    or not self.dag_prev_dict[frontier].isdisjoint(
                        set.union(set(segment),
                                  *[self.dag_prev_dict[i] for i in segment]))

            # The previous vertices in the current segment, whose output is
            # still on-chip and has not stored into memory.
            pending_prevs = self.dag_prev_dict[frontier] - done
            assert pending_prevs.issubset(segment)

            if not share_deps or len(pending_prevs) > 1:
                # Not sharing any dependencies (rule 1), or more than one
                # previous vertices in the current segment (rule 2).

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
                yield seg_idx, segment

                # Skip if have done.
                if no_repeating and frontier + 1 in self.seg_vertex_done:
                    continue

                # Recursion.
                for tpl in self._gen_segment(seg_idx + 1, frontier + 1,
                                             done.union(segment),
                                             no_repeating=no_repeating):
                    yield tpl

        if no_repeating:
            assert vertex_idx not in self.seg_vertex_done
            self.seg_vertex_done.add(vertex_idx)

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

