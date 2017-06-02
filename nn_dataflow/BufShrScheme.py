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

import itertools

from . import DataCategoryEnum as de
from . import ParallelEnum as pe
from . import Util
from .PhyDim2 import PhyDim2
from .Util import StringifyClass

class BufShrScheme(StringifyClass):
    '''
    The buffer sharing scheme.
    '''

    def __init__(self, part):

        # Dimension of the node group.
        self.dims = [PhyDim2(float('nan'), float('nan'))] * de.NUM
        # Distance between the neighbors in the node group.
        self.nbr_dists = [PhyDim2(float('nan'), float('nan'))] * de.NUM

        # Fil weights in gbuf can be shared by nodes with OFMP and BATP.
        idx_ofmp = part.order.index(pe.OFMP)
        idx_batp = part.order.index(pe.BATP)
        dim_ofmp = part.dim(pe.OFMP)
        dim_batp = part.dim(pe.BATP)
        # If only one of OFMP and BATP exists, use that one.
        if dim_ofmp.size() == 1:
            self.dims[de.FIL] = dim_batp
            self.nbr_dists[de.FIL] = part.part_neighbor_dist(pe.BATP)
        elif dim_batp.size() == 1:
            self.dims[de.FIL] = dim_ofmp
            self.nbr_dists[de.FIL] = part.part_neighbor_dist(pe.OFMP)
        else:
            # If both exist ...
            if abs(idx_ofmp - idx_batp) == 1:
                # ... and are adjacent in the partitioning hierarchy, use
                # both.
                self.dims[de.FIL] = dim_batp * dim_ofmp
                self.nbr_dists[de.FIL] = part.part_neighbor_dist(
                    pe.OFMP if idx_ofmp > idx_batp else pe.BATP)
            else:
                # ... but are not adjacent, use the bottom one (with
                # smaller distance).
                if idx_ofmp > idx_batp:
                    self.dims[de.FIL] = dim_ofmp
                    self.nbr_dists[de.FIL] = part.part_neighbor_dist(pe.OFMP)
                else:
                    self.dims[de.FIL] = dim_batp
                    self.nbr_dists[de.FIL] = part.part_neighbor_dist(pe.BATP)

        # Ifmaps in gbuf can be shared by nodes with OUTP.
        self.dims[de.IFM] = part.dim(pe.OUTP)
        self.nbr_dists[de.IFM] = part.part_neighbor_dist(pe.OUTP)

        # Ofmaps in gbuf can be shared by nodes with INNP.
        self.dims[de.OFM] = part.dim(pe.INPP)
        self.nbr_dists[de.OFM] = part.part_neighbor_dist(pe.INPP)

    def dim(self, dce):
        ''' Get the buffer sharing node group dimensions. '''
        return self.dims[dce]

    def size(self, dce):
        ''' Get the buffer sharing node group size. '''
        return self.dims[dce].size()

    def nhops_rotate_all(self, dce, subgrp_size):
        '''
        Number of hops for rotation operation of an entire round.

        The number of hops is relative to the total unique data size. E.g.,
        when the data are in N nodes and each node has 1/M data, if all the
        data have been transferred by 1 hop, the number of hops is N / M.

        The data are spread in N nodes, where N is the group size. Each node
        holds 1/M data, where M is given by `subgrp_size`. M is rounded up to a
        factor of N, M' >= M, and each M' nodes is a subgroup. There are N//M'
        == N//M subgroups. If M' == M, there are no redundant data in the nodes
        of a subgroup.

        Rotation means the following operation: nodes exchange their data with
        the minimum number of hops, until every node has seen all the data.

        How to rotate:

        Each subgroup rotate their data independently. A subgroup is typically
        2D. We chain the nodes in a snaking fashion with a priority dimension.
        E.g., if the priority dimension is H (the 1st one), then the node chain
        is (0,0), (1,0), ..., (H-1,0), (H-1,1), (H-2,1), ..., (0,1), (0,2),
        ..., i.e., first go along H to the end, then turn to W and go one hop
        to the next H, then turn and go long H, etc.. The priority dimension is
        chosen to minimize the overall rotation hops.

        We store data in the chained M' nodes of a subgroup as follow, where
        the index is the i-th 1/M chunk:

        M-1, M-2, ..., 1, 0, | M-1, M-2, ..., 2M-M'

        The first M nodes loop their data over. In addition, the (M-1)-th node
        also sends its data to the M-th node. The last M'-M nodes sequentially
        send data to the right side, and the last node does not send data.

        So in the next step:

        0, M-1, ..., 2, 1, | 0, M-1, ..., 2M-M'+1

        And so on until the last step:

        M-2, M-3, ..., 0, M-1, | M-2, M-3, ..., 2M-M'-1

        Overall, each node except for the last one sends its 1/M data to the
        right neighbor at each of the M-1 step. And the (M-1)-th node also
        sends its 1/M data to the 0-th node.
        '''

        subgrp_dim, idx_pr = self._subgrp_dim(dce, subgrp_size)

        # 1. Send to right neighbor.
        # If H < W, rotate along H dimension, i.e., go along H to the end, then
        # turn to W and go one hop to the next H, then turn and go long H, ...
        d_pr = subgrp_dim[idx_pr]
        d_npr = subgrp_dim[1 - idx_pr]
        dist_pr = self.nbr_dists[dce][idx_pr]
        dist_npr = self.nbr_dists[dce][1 - idx_pr]
        # Per-step nhops = (H-1) * W * Dh + (W-1) * Dw
        nhops_nbr = (d_pr - 1) * d_npr * dist_pr + (d_npr - 1) * dist_npr

        # 2. (M-1)-th node loops back to the 0-th node.
        # Position of the (M-1)-th node.
        coord = self._coordinate(subgrp_size - 1, subgrp_dim, idx_pr)
        # Per-step nhops = distance back to the 0-th node.
        nhops_lpbk = sum(coord * self.nbr_dists[dce])

        # All steps; normalize; all subgroups.
        nhops = (nhops_nbr + nhops_lpbk) \
                * (subgrp_size - 1) \
                * (1. / subgrp_size) \
                * (self.size(dce) // subgrp_size)

        return nhops

    def nhops_broadcast(self, dce, subgrp_size):
        '''
        Number of hops for broadcast operation.

        The number of hops is relative to the total unique data size. E.g.,
        when the data are in N nodes and each node has 1/M data, if all the
        data have been transferred by 1 hop, the number of hops is N / M.

        Broadcast means the following operation: initially each node holds 1/N
        data, where N is the group size. Finally each node holds 1/M data,
        where M is given by `subgrp_size`. The resulting state should match the
        initial rotation state. See the rotation function about how the data
        are distributed.
        '''

        subgrp_dim, idx_pr = self._subgrp_dim(dce, subgrp_size)
        dim = self.dim(dce)
        size = self.size(dce)

        # Data index means which 1/M data; source index means which of the N
        # nodes initially.

        # Each element is a list of pairs for idx_data, where each pair is
        # (idx_src, p), where p is the amount of idx_data data from this
        # idx_src node.
        data_src_list = [[] for _ in range(subgrp_size)]

        idx_src = 0
        for idx_data in range(subgrp_size):
            # idx_src = idx_data / M * N
            beg_src = idx_data * 1. / subgrp_size * size
            end_src = (idx_data + 1) * 1. / subgrp_size * size
            assert idx_src + 1 > beg_src - 1e-5

            # E.g., (3.4, 5.2) will lead to [(3, 0.6), (4, 1), (5, 0.2)].
            cbeg = beg_src
            while True:
                cend = min(end_src, idx_src + 1.)
                pair = (idx_src, (cend - cbeg) / size)
                data_src_list[idx_data].append(pair)
                if idx_src + 1 > end_src - 1e-5:
                    break
                idx_src += 1
                cbeg = cend

        # Check total amount of source data.
        _total = sum(sum(pair[1] for pair in dsrc) for dsrc in data_src_list)
        assert Util.isclose(_total, 1., abs_tol=1e-3)
        # Check per source amount of data.
        _per_src = [0.] * size
        for dsrc in data_src_list:
            for pair in dsrc:
                _per_src[pair[0]] += pair[1]
        assert all(Util.isclose(p, 1. / size, rel_tol=1e-3) for p in _per_src)

        nhops = 0

        for sgch, sgcw in itertools.product(range(dim.h // subgrp_dim.h),
                                            range(dim.w // subgrp_dim.w)):
            c_base = PhyDim2(h=sgch * subgrp_dim.h, w=sgcw * subgrp_dim.w)
            for index in range(subgrp_dim.size()):
                c_offset = self._coordinate(index, subgrp_dim, idx_pr)
                # Absolute coordinate of destination node in group.
                c_data = c_base + c_offset

                for pair in data_src_list[index % subgrp_size]:
                    # Absolute coordinate of source node in group. Use the same
                    # priority dimension as the subgroup.
                    c_src = self._coordinate(pair[0], dim, idx_pr)

                    # Distance between source and destination.
                    dist = sum(abs(dst - src) * d
                               for src, dst, d in zip(c_src, c_data,
                                                      self.nbr_dists[dce]))

                    nhops += pair[1] * dist

        return nhops

    def _subgrp_dim(self, dce, subgrp_size):
        '''
        Decide the subgroup dimensions and the priority dimension index.
        Priority dimension is the one along which rotation happens.
        '''
        # Round up subgroup size to a factor of the group size.
        true_subgrp_size = subgrp_size
        size = self.size(dce)
        while size % true_subgrp_size:
            true_subgrp_size += 1
            if true_subgrp_size > size:
                raise ValueError('BufShrScheme: subgroup is larger than group. '
                                 '{} vs. {}.'.format(subgrp_size, size))

        dim = self.dim(dce)
        nbr_dist = self.nbr_dists[dce]

        # The dimension with smaller/larger distance.
        idx_sm = 0 if nbr_dist[0] <= nbr_dist[1] else 1
        idx_lg = 1 - idx_sm
        dim_sm = dim[idx_sm]

        # The smaller-distance dimension is the priority dimension.
        idx_pr = idx_sm

        tpl = [1] * 2

        # We try to use as much as possible from the smaller-distance dimension
        # to the subgroup. Figure out the maximum factor.
        for f, _ in Util.factorize(dim_sm, 2):
            if f > tpl[idx_sm] and true_subgrp_size % f == 0:
                tpl[idx_sm] = f

        tpl[idx_lg] = true_subgrp_size // tpl[idx_sm]

        subgrp_dim = PhyDim2(*tpl)
        assert subgrp_dim.size() == true_subgrp_size

        return subgrp_dim, idx_pr

    @staticmethod
    def _coordinate(index, dim, idx_pr):
        '''
        The coordinate of a node with sequential index `index` in the 2D nodes
        with dimensions `dim`.  The index increases first along the priority
        dimension given by `idx_pr` as the dimension index. Return a PhyDim2
        coordinate.
        '''
        dim_pr, dim_npr = dim if idx_pr == 0 else reversed(dim)
        coord_npr, coord_pr = divmod(index, dim_pr)
        assert coord_npr < dim_npr and coord_pr < dim_pr
        # We go backward in the odd H, i.e., snaking.
        if coord_npr % 2 == 1:
            coord_pr = dim_pr - 1 - coord_pr
        coord = PhyDim2(coord_pr, coord_npr) if idx_pr == 0 \
                else PhyDim2(coord_npr, coord_pr)
        return coord

