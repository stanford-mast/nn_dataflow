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
from .DataLayout import DataLayout
from .FmapRange import FmapPosition, FmapRange, FmapRangeMap
from .Layer import ConvLayer, LocalRegionLayer
from .PartitionScheme import PartitionScheme
from .PhyDim2 import PhyDim2

'''
Parallel process partitioning.

Partition among multiple copies of the PE arrays.

For our case, only deal with up to 2D layout of PE arrays.
'''

def gen_partition(layer, batch_size, dim_nodes, options):
    '''
    Generator for all possible partitioning schemes that partition `layer` into
    2D `dim_nodes` nodes.
    '''
    for ph, pw in itertools.product(Util.factorize(dim_nodes.h, pe.NUM),
                                    Util.factorize(dim_nodes.w, pe.NUM)):

        pdims = [PhyDim2(h, w) for h, w in zip(ph, pw)]

        # Batch partitoning.
        if (not options.partition_batch) and pdims[pe.BATP].size() > 1:
            continue
        elif batch_size % pdims[pe.BATP].size() != 0:
            continue

        if options.partition_hybrid:
            # Require partition is approximately dividable of total size.
            if not Util.approx_dividable(layer.nofm, pdims[pe.OUTP].size()):
                continue
            if not Util.approx_dividable(layer.hofm, pdims[pe.OFMP].h) \
                    or not Util.approx_dividable(layer.wofm, pdims[pe.OFMP].w):
                continue

            if (not options.partition_ifmaps) and pdims[pe.INPP].size() > 1:
                continue
            else:
                if isinstance(layer, ConvLayer):
                    if not Util.approx_dividable(layer.nifm,
                                                 pdims[pe.INPP].size()):
                        continue
                elif isinstance(layer, LocalRegionLayer):
                    if pdims[pe.INPP].size() > 1:
                        continue
        else:
            assert not options.partition_ifmaps
            if pdims[pe.INPP].size() != 1:
                continue

            if layer.hofm == 1 and layer.wofm == 1:
                # FC layer: no OFMP.
                if pdims[pe.OFMP].size() != 1:
                    continue
            else:
                # CONV layer: no OUTP.
                if pdims[pe.OUTP].size() != 1:
                    continue

        # For different order.
        for order in itertools.permutations(tuple(range(pe.NUM))):
            # Size-(1, 1) partition has no effect, so its order is not
            # relevant. Force them at the beginning.
            no_part = [v for v in range(pe.NUM) if pdims[v].size() == 1]
            if not all([order[i] == no_part[i] for i in range(len(no_part))]):
                continue

            # Batch parallelism should be at the top.
            if pe.BATP not in no_part and order[len(no_part)] != pe.BATP:
                continue

            part = PartitionScheme(order, pdims)
            assert part.dim() == dim_nodes

            yield part


def part_layer_ofmap_range(layer, batch_size, part, pidx):
    '''
    Get the partitioned ofmap range including batch, for the given partition
    index `pidx`.
    '''
    # Batch partition.
    idx_bat = pidx[pe.BATP].h * part.dim(pe.BATP).w + pidx[pe.BATP].w
    b_beg, b_end = Util.get_ith_range((0, batch_size),
                                      idx_bat, part.size(pe.BATP))

    # Ofmap channel partition.
    idx_ofm = pidx[pe.OUTP].h * part.dim(pe.OUTP).w + pidx[pe.OUTP].w
    n_beg, n_end = Util.get_ith_range((0, layer.nofm),
                                      idx_ofm, part.size(pe.OUTP))

    # Fmap height tiling.
    h_beg, h_end = Util.get_ith_range((0, layer.hofm),
                                      pidx[pe.OFMP].h, part.dim(pe.OFMP).h)

    # Fmap width tiling.
    w_beg, w_end = Util.get_ith_range((0, layer.wofm),
                                      pidx[pe.OFMP].w, part.dim(pe.OFMP).w)

    return FmapRange(FmapPosition(b=b_beg, n=n_beg, h=h_beg, w=w_beg),
                     FmapPosition(b=b_end, n=n_end, h=h_end, w=w_end))


def part_layer_ifmap_range(layer, batch_size, part, pidx):
    '''
    Get the partitioned ifmap range including batch, for the given partition
    index `pidx`.
    '''

    ofmap_range = part_layer_ofmap_range(layer, batch_size, part, pidx)
    b_orng, n_orng, h_orng, w_orng = ofmap_range.beg_end('b', 'n', 'h', 'w')

    # Batch partition.
    b_beg, b_end = b_orng

    if isinstance(layer, ConvLayer):
        # Ifmap channel partition.
        idx_ifm = pidx[pe.INPP].h * part.dim(pe.INPP).w + pidx[pe.INPP].w
        n_beg, n_end = Util.get_ith_range((0, layer.nifm),
                                          idx_ifm, part.size(pe.INPP))
        # Fmap height tiling.
        h_beg, h_end = h_orng
        # xy_i = xy_o * stride + (0 ... sfil-1)
        h_beg = h_beg * layer.htrd
        h_end = max(h_beg, (h_end - 1) * layer.htrd + layer.sfil)

        # Fmap width tiling.
        w_beg, w_end = w_orng
        w_beg = w_beg * layer.wtrd
        w_end = max(w_beg, (w_end - 1) * layer.wtrd + layer.sfil)

    elif isinstance(layer, LocalRegionLayer):
        # Ifmap channel partition.
        n_beg, n_end = n_orng
        n_beg = max(0, n_beg - layer.nreg // 2)
        n_end = min(layer.nifm, n_end + layer.nreg - layer.nreg // 2)

        # Fmap height tiling.
        h_beg, h_end = h_orng
        h_beg = max(0, h_beg * layer.htrd - layer.hreg // 2)
        h_end = min(layer.hifm,
                    h_end * layer.htrd + layer.hreg - layer.hreg // 2)

        # Fmap width tiling.
        w_beg, w_end = w_orng
        w_beg = max(0, w_beg * layer.wtrd - layer.wreg // 2)
        w_end = min(layer.wifm,
                    w_end * layer.wtrd + layer.wreg - layer.wreg // 2)

    assert n_end <= layer.nifm and h_end <= layer.hifm and w_end <= layer.wifm

    return FmapRange(FmapPosition(b=b_beg, n=n_beg, h=h_beg, w=w_beg),
                     FmapPosition(b=b_end, n=n_end, h=h_end, w=w_end))


def part_layer_unit_nhops(layer, batch_size, part, filter_node_coord_list,
                          ifmap_layout, ofmap_layout, options):
    '''
    Get total number of hops for each data category when partitioning the given
    layer computation workload with PartitionScheme `part`.

    `ifmap_layout` and (optional) `ofmap_layout` specify the data layouts of
    the i/ofmaps in memory as FmapRangeMap instances, mapping FmapPosition to
    node coordinate. The node coordinate is relative to the origin of the
    computation node region.

    Since the filters are read-only and independent of the previous layer
    computation, we can duplicate filters in multiple memory nodes given by
    `filter_node_coord_list`, and assume the accesses can be forwarded to the
    nearest memory.

    Return a tuple with each element being the number of hops for each data
    category.
    '''

    nhops = [0] * de.NUM

    del options

    pidx_mid_inpp = PhyDim2(h=part.dim(pe.INPP).h//2,
                            w=part.dim(pe.INPP).w//2)

    for pidx in part.gen_pidx():
        coord = part.coordinate(pidx)

        # Computation workload (as an ofmap range) of this node coordinate.
        ofrng = part_layer_ofmap_range(layer, batch_size, part, pidx)

        # Required ifmap range.
        ifrng = part_layer_ifmap_range(layer, batch_size, part, pidx)

        # Ifmap access.
        nhops[de.IFM] += ifmap_layout.total_transfer_nhops(ifrng, coord)

        # Ofmap access.
        # Additional synchronization is necessary between INPP nodes. Only one
        # node (the mid one) fetch the previously-partial-accumulated data from
        # memory into buffers and start on it. Other nodes start on zero and
        # send the results to the mid node to accumulate there.
        if pidx[pe.INPP] == pidx_mid_inpp:
            # The mid node. Fetch from memory
            nhops[de.OFM] += ofmap_layout.total_transfer_nhops(ofrng, coord)
        else:
            # Others. Send to the mid node (one way).
            pidx_mid = list(pidx)
            pidx_mid[pe.INPP] = pidx_mid_inpp
            dist = coord.hop_dist(part.coordinate(pidx_mid))
            nhops[de.OFM] += ofrng.size() * dist / 2  # half because one way.

        # filter access.
        if isinstance(layer, ConvLayer):
            fil_size = ofrng.size('n') * ifrng.size('n') * layer.filter_size()
            min_hops = min(coord.hop_dist(c) for c in filter_node_coord_list)
            nhops[de.FIL] += fil_size * min_hops

    return nhops


def get_ofmap_layout(layer, batch_size, part, output_mem_region):
    '''
    Decide the ofmap data layout as a DataLayout instance, given the
    PartitionScheme `part` of the computation workloads and the memory
    NodeRegion `output_mem_region`.

    The ofmap partitioning is calculated by shrinking or extending the
    computation partitioning, while trying to maintain the same layout shape.
    '''

    dim_part = part.dim()
    dim_omr = output_mem_region.dim

    if dim_omr.size() == 0:
        raise ValueError('Partition ofmap: empty node region.')

    # Start with the same as computation partitioning.
    ofmap_order = part.order
    ofmap_pdims = [list(dim) for dim in part.pdims]
    # Adjust each dimension.
    for di in range(2):
        pd = dim_part[di]
        od = dim_omr[di]

        if od > pd:
            # Ofmap dimension > computation dimension. Extend.
            ext = od // pd
            # Apply the extension to the top level.
            ofmap_pdims[ofmap_order[0]][di] *= ext
        else:
            # Computation dimension >= ofmap dimension, shrink.
            # Go from bottom to top. Keep bottom (first) levels unchanged, and
            # shrink top (latter) levels.
            for pae in reversed(ofmap_order):
                if od > ofmap_pdims[pae][di]:
                    # Remaining size in ofmap dimension is enough for current
                    # level. Use it to keep the current level the same size.
                    od //= ofmap_pdims[pae][di]
                else:
                    # Remaining size in ofmap dimension is not enough. Shrink
                    # current level to be whatever remains.
                    ofmap_pdims[pae][di] = od
                    od = 1
    # Eliminate the partitioning schemes which do not partition ofmaps (i.e.,
    # correspond to the same ofmap range). Merge them to BATP.
    for pae in range(pe.NUM):
        if pae not in [pe.OUTP, pe.OFMP, pe.BATP]:
            ofmap_pdims[pe.BATP] = [x * y for x, y in zip(ofmap_pdims[pe.BATP],
                                                          ofmap_pdims[pae])]
            ofmap_pdims[pae] = [1, 1]

    ofmap_part = PartitionScheme(order=ofmap_order, pdims=ofmap_pdims)
    assert all(od <= omrd for od, omrd in zip(ofmap_part.dim(), dim_omr)), \
            'Partition ofmap: ofmap partitioning {} is invalid within ' \
            'memory region {}.'.format(ofmap_part, str(output_mem_region))

    # Make layout.
    ofmap_frmap = FmapRangeMap()
    for pidx in ofmap_part.gen_pidx():
        frng = part_layer_ofmap_range(layer, batch_size, ofmap_part, pidx)
        coord = ofmap_part.coordinate(pidx)
        ofmap_frmap.add(frng, (coord,))

    ofmap_layout = DataLayout(frmap=ofmap_frmap,
                              origin=output_mem_region.origin)

    return ofmap_layout

