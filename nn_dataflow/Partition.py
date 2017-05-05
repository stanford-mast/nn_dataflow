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
from .FmapRange import FmapRangeMap
from .Layer import ConvLayer
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
        else:
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

    If `ofmap_layout` is None, the ofmaps are stored to the memory of the same
    node of the computation, which results in no hops for ofmaps.

    Return a tuple with each element being the number of hops for each data
    category.
    '''

    nhops = [0] * de.NUM

    del options

    for pidx in part.gen_pidx():
        coord = part.coordinate(pidx)

        # Computation workload (as an ofmap range) of this node coordinate.
        frng = part.part_fmap_range(
            batch_size, layer.nofm, layer.hofm, layer.wofm, pidx)

        # Required ifmap range.
        frng_src = frng.corresponding_input_fmap_range(layer)

        # ifmap access.
        nhops[de.IFM] += ifmap_layout.total_transfer_nhops(frng_src, coord)

        # ofmap access.
        if ofmap_layout is not None:
            nhops[de.OFM] += ofmap_layout.total_transfer_nhops(frng, coord)

        # filter access.
        if isinstance(layer, ConvLayer):
            fil_size = frng.size('n') * frng_src.size('n') * layer.filter_size()
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
    ofmap_part = PartitionScheme(order=ofmap_order, pdims=ofmap_pdims)
    assert all(od <= omrd for od, omrd in zip(ofmap_part.dim(), dim_omr)), \
            'Partition ofmap: ofmap partitioning {} is invalid within ' \
            'memory region {}.'.format(ofmap_part, str(output_mem_region))

    # Make layout.
    ofmap_frmap = FmapRangeMap()
    for pidx in ofmap_part.gen_pidx():
        frng = ofmap_part.part_fmap_range(batch_size, layer.nofm, layer.hofm,
                                          layer.wofm, pidx)
        coord = ofmap_part.coordinate(pidx)
        ofmap_frmap.add(frng, (coord,))

    ofmap_layout = DataLayout(frmap=ofmap_frmap,
                              origin=output_mem_region.origin)

    return ofmap_layout

