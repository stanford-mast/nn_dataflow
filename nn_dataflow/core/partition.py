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

from . import data_category_enum as de
from . import parallel_enum as pe
from .. import util
from .data_layout import DataLayout
from .fmap_range import FmapPosition, FmapRange, FmapRangeMap
from .layer import ConvLayer, LocalRegionLayer
from .partition_scheme import PartitionScheme
from .phy_dim2 import PhyDim2

'''
Parallel process partitioning.

Partition among multiple copies of the PE arrays.

For our case, only deal with up to 2D layout of PE arrays.
'''

def gen_partition(layer, batch_size, dim_nodes, options, guaranteed=False):
    '''
    Generator for all possible partitioning schemes that partition `layer` into
    2D `dim_nodes` nodes.

    If `guaranteed` is True, we guarantee to yield at least one partitioning
    scheme regardless of efficiency.
    '''
    # pylint: disable=too-many-branches

    yielded = False

    for ph, pw in itertools.product(util.factorize(dim_nodes.h, pe.NUM),
                                    util.factorize(dim_nodes.w, pe.NUM)):

        pdims = [PhyDim2(h, w) for h, w in zip(ph, pw)]

        # Batch partitoning.
        if (not options.partition_batch) and pdims[pe.BATP].size() > 1:
            continue
        elif batch_size % pdims[pe.BATP].size() != 0:
            continue

        if options.partition_hybrid:
            # Require partition is approximately dividable of total size.
            if not util.approx_dividable(layer.nofm, pdims[pe.OUTP].size()):
                continue
            if not util.approx_dividable(layer.hofm, pdims[pe.OFMP].h) \
                    or not util.approx_dividable(layer.wofm, pdims[pe.OFMP].w):
                continue

            if (not options.partition_ifmaps) and pdims[pe.INPP].size() > 1:
                continue
            else:
                if isinstance(layer, ConvLayer):
                    if not util.approx_dividable(layer.nifm,
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

        # Skip the transpose equivalence. Only partitioning scheme without OFMP
        # and with only 1-D partitioning could have equivalence, since we
        # always index in height-major order.
        if pdims[pe.OFMP].size() == 1 \
                and all(pd.h == 1 or pd.w == 1 for pd in pdims):
            pdhs, pdws = zip(*pdims)
            if pdhs > pdws:
                continue

        # For different order.
        for order in itertools.permutations(range(pe.NUM)):

            # Partition with dim-1 has no effect, so its order is not relevant.
            skip = False
            for idx in range(pe.NUM - 1):
                pae1 = order[idx]
                pae2 = order[idx + 1]
                pdim1 = pdims[pae1]
                pdim2 = pdims[pae2]

                # Invalid cases include:
                # - both are (1, 1) but not in order of ParallelEnum.
                # - (1, 1) after non-(1, 1).
                # - (1, non-1) after (non-1, 1) of not BATP.
                if pdim1.size() == 1 and pdim2.size() == 1 and pae1 > pae2:
                    skip = True
                    break
                if pdim1.size() > 1 and pdim2.size() == 1:
                    skip = True
                    break
                if pae1 != pe.BATP and pdim2.h == 1 and pdim2.w > 1 \
                        and pdim1.h > 1 and pdim1.w == 1:
                    skip = True
                    break
            if skip:
                continue

            no_part = [pae for pae in range(pe.NUM) if pdims[pae].size() == 1]
            # Batch parallelism should be at the top.
            if pe.BATP not in no_part and order[len(no_part)] != pe.BATP:
                continue

            part = PartitionScheme(order, pdims)
            assert part.dim() == dim_nodes

            yield part

            yielded = True

    if guaranteed and not yielded:
        # None of the Partitioning schemes are valid. May be due to
        # non-dividability. Return a single naive scheme, with only OFMP or
        # only OUTP.

        pdims = [PhyDim2(1, 1)] * pe.NUM
        order = range(pe.NUM)

        if layer.hofm == 1 and layer.wofm == 1:
            # Only OUTP, no OFMP.
            pdims[pe.OUTP] = dim_nodes
        else:
            # Only OFMP, no OUTP.
            pdims[pe.OFMP] = dim_nodes

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
    b_beg, b_end = util.get_ith_range((0, batch_size),
                                      idx_bat, part.size(pe.BATP))

    # Ofmap channel partition.
    idx_ofm = pidx[pe.OUTP].h * part.dim(pe.OUTP).w + pidx[pe.OUTP].w
    n_beg, n_end = util.get_ith_range((0, layer.nofm),
                                      idx_ofm, part.size(pe.OUTP))

    # Fmap height tiling.
    h_beg, h_end = util.get_ith_range((0, layer.hofm),
                                      pidx[pe.OFMP].h, part.dim(pe.OFMP).h)

    # Fmap width tiling.
    w_beg, w_end = util.get_ith_range((0, layer.wofm),
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
        n_beg, n_end = util.get_ith_range((0, layer.nifm),
                                          idx_ifm, part.size(pe.INPP))
        # Fmap height tiling.
        h_beg, h_end = h_orng
        # xy_i = xy_o * stride + (0 ... sfil-1)
        h_beg = h_beg * layer.htrd
        h_end = max(h_beg, (h_end - 1) * layer.htrd + layer.hfil)

        # Fmap width tiling.
        w_beg, w_end = w_orng
        w_beg = w_beg * layer.wtrd
        w_end = max(w_beg, (w_end - 1) * layer.wtrd + layer.wfil)

    elif isinstance(layer, LocalRegionLayer):
        # Ifmap channel partition.
        n_beg, n_end = n_orng
        n_beg = max(0, n_beg - layer.nreg // 2)
        n_end = min(layer.nifm, n_end + layer.nreg - layer.nreg // 2)

        # Fmap height tiling.
        h_beg, h_end = h_orng
        h_beg = h_beg * layer.htrd
        h_end = max(h_beg, (h_end - 1) * layer.htrd + layer.hreg)

        # Fmap width tiling.
        w_beg, w_end = w_orng
        w_beg = w_beg * layer.wtrd
        w_end = max(w_beg, (w_end - 1) * layer.wtrd + layer.wreg)

    assert n_end <= layer.nifm and h_end <= layer.hifm and w_end <= layer.wifm

    return FmapRange(FmapPosition(b=b_beg, n=n_beg, h=h_beg, w=w_beg),
                     FmapPosition(b=b_end, n=n_end, h=h_end, w=w_end))


def part_layer_unit_nhops(layer, batch_size, part, node_region,
                          filter_nodes, ifmap_layout, ofmap_layout, options):
    '''
    Get total number of hops for each data category when partitioning the given
    layer computation workload with PartitionScheme `part` on NodeRegion
    `node_region`.

    `ifmap_layout` and (optional) `ofmap_layout` specify the data layouts of
    the i/ofmaps in memory as FmapRangeMap instances, mapping FmapPosition to
    node coordinate. The node coordinate is relative to the origin of the
    computation node region.

    Since the filters are read-only and independent of the previous layer
    computation, we can duplicate filters in multiple memory nodes given by
    `filter_nodes`, and assume the accesses can be forwarded to the nearest
    memory.

    All node coordinates are given as absolute coordinates.

    Return a tuple with each element being the number of hops for each data
    category.
    '''

    del options

    # FmapRange --> coordinates that need this data.
    fil_dict = {}
    ofm_dict = {}
    ifm_dict = {}

    for pidx in part.gen_pidx():
        coord = part.coordinate(node_region, pidx)

        # Computation workload (as an ofmap range) of this node coordinate.
        ofrng = part_layer_ofmap_range(layer, batch_size, part, pidx)
        if ofrng.size() > 0:
            ofm_dict.setdefault(ofrng, []).append(coord)

        # Required ifmap range.
        ifrng = part_layer_ifmap_range(layer, batch_size, part, pidx)
        if ifrng.size() > 0:
            ifm_dict.setdefault(ifrng, []).append(coord)

        # Filters, as a tuple of ((i_beg, i_end), (o_beg, o_end)).
        filrng = tuple(ifrng.beg_end('n')) + tuple(ofrng.beg_end('n'))
        if filrng[0][1] > filrng[0][0] and filrng[1][1] > filrng[1][0]:
            fil_dict.setdefault(filrng, []).append(coord)

    if isinstance(layer, ConvLayer):
        assert all(len(v) == part.size(pe.INPP) for v in ofm_dict.values()), \
                '{}\n{}'.format(part.size(pe.INPP),
                                [(str(k), str(v)) for k, v in ofm_dict.items()])
        assert all(len(v) == part.size(pe.OUTP) for v in ifm_dict.values()), \
                '{}\n{}'.format(part.size(pe.OUTP),
                                [(str(k), str(v)) for k, v in ifm_dict.items()])
        assert all(len(v) == part.size(pe.OFMP, pe.BATP)
                   for v in fil_dict.values()), \
                '{}\n{}'.format(part.size(pe.OFMP, pe.BATP),
                                [(str(k), str(v)) for k, v in fil_dict.items()])

    nhops = [0] * de.NUM

    # Ifmap access.
    for ifrng, coord_list in ifm_dict.items():
        nhops[de.IFM] += sum(ifmap_layout.total_transfer_nhops(ifrng, coord)
                             for coord in coord_list)

    # Ofmap access.
    # Additional synchronization is necessary between INPP nodes. Only one node
    # (the mid one) fetch the previously-partial-accumulated data from memory
    # into buffers and start on it. Other nodes start on zero and send the
    # results to the mid node to accumulate there.
    for ofrng, coord_list in ofm_dict.items():
        mid_idx = len(coord_list) // 2
        for idx, coord in enumerate(coord_list):
            if idx == mid_idx:
                # The mid node. Fetch from memory
                nhops[de.OFM] += ofmap_layout.total_transfer_nhops(ofrng, coord)
            else:
                # Others. Send to the mid node (one way).
                # The total fetch times (reads and writes) of OFM is f = 2n - 1
                # (no read for the first time), i.e., n - 1 reads and n writes.
                # Only writes need to send to the mid node, i.e., (f + 1) / 2
                # rather than f times, approximately half.
                dist = coord.hop_dist(coord_list[mid_idx])
                nhops[de.OFM] += ofrng.size() * dist / 2

    # Filter access.
    if isinstance(layer, ConvLayer):
        for filrng, coord_list in fil_dict.items():
            fil_size = (filrng[0][1] - filrng[0][0]) \
                    * (filrng[1][1] - filrng[1][0]) \
                    * layer.filter_size()
            for coord in coord_list:
                min_hops = min(coord.hop_dist(c)
                               for c in filter_nodes)
                nhops[de.FIL] += fil_size * min_hops

    return nhops


def get_ofmap_layout(layer, batch_size, part, out_data_region):
    '''
    Decide the ofmap data layout as a DataLayout instance, given the
    PartitionScheme `part` of the computation workloads and the output data
    NodeRegion `out_data_region`.

    The ofmap partitioning is calculated by shrinking or extending the
    computation partitioning, while trying to maintain the same layout shape.
    '''
    # Only work on the partitioning schemes related to ofmaps.
    ofmap_paes = [pe.OUTP, pe.OFMP, pe.BATP]
    part = PartitionScheme(order=part.order,
                           pdims=[part.dim(pae) if pae in ofmap_paes
                                  else PhyDim2(1, 1) for pae in range(pe.NUM)])

    dim_part = part.dim()
    dim_omr = out_data_region.dim

    if dim_omr.size() == 0:
        raise ValueError('partition ofmap: empty node region.')

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
            top_pae = next(pae for pae in ofmap_order if pae in ofmap_paes)
            ofmap_pdims[top_pae][di] *= ext
        else:
            # Computation dimension >= ofmap dimension, shrink.
            # Go from bottom to top. Keep bottom (first) levels unchanged, and
            # shrink top (latter) levels.
            for pae in reversed(ofmap_order):
                if od >= ofmap_pdims[pae][di]:
                    # Remaining size in ofmap dimension is enough for current
                    # level. Use it to keep the current level the same size.
                    od //= ofmap_pdims[pae][di]
                else:
                    # Remaining size in ofmap dimension is not enough. Shrink
                    # current level to be whatever remains.
                    ofmap_pdims[pae][di] = od
                    od = 1

    ofmap_part = PartitionScheme(order=ofmap_order, pdims=ofmap_pdims)
    assert all(od <= omrd for od, omrd in zip(ofmap_part.dim(), dim_omr)), \
            'partition ofmap: ofmap partitioning {} is invalid within ' \
            'memory region {}.'.format(ofmap_part, str(out_data_region))

    # Make layout.
    ofmap_frmap = FmapRangeMap()
    for pidx in ofmap_part.gen_pidx():
        frng = part_layer_ofmap_range(layer, batch_size, ofmap_part, pidx)
        coord = ofmap_part.coordinate(out_data_region, pidx)
        ofmap_frmap.add(frng, (coord,))

    ofmap_layout = DataLayout(frmap=ofmap_frmap,
                              origin=PhyDim2(0, 0),
                              type=out_data_region.type)
    assert ofmap_layout.is_in_region(out_data_region)

    return ofmap_layout

