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
from .FmapRange import FmapRangeMap
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


def part_layer_unit_nhops(layer, batch_size, part, part_src, offset_src,
                          part_dst, offset_dst, options):
    '''
    Get total number of hops for each data category when partitioning the given
    layer with `part` and partitioning the source (previous layer) with
    `part_src`. The node region origin offset of src is `offset_src`. In
    addition, optionally (set to None if not used), the destination (next layer
    or memory storage for current layer) is partitioned with `part_dst`, and
    the node region origin offset of dst is `offset_dst`.

    Return a tuple with each element being the number of hops for each data
    category.
    '''

    nhops = [0] * de.NUM

    del options

    # Prepare mapping from FmapPosition to coordinate for src. Coordinate is
    # translated to current origin.
    fp2c_src = FmapRangeMap()
    for pidx in part_src.gen_pidx():
        coord = part_src.coordinate(pidx)
        frng = part_src.part_fmap_range(
            batch_size, layer.nifm, layer.hifm, layer.wifm, pidx)
        fp2c_src.add(frng, coord + offset_src)

    # Prepare mapping from FmapPosition to coordinate for dst. Coordinate is
    # translated to current origin.
    if part_dst is None:
        # Set to be same as layer partition if None.
        part_dst = part
        offset_dst = PhyDim2(0, 0)
    fp2c_dst = FmapRangeMap()
    for pidx in part_dst.gen_pidx():
        coord = part_dst.coordinate(pidx)
        frng = part_dst.part_fmap_range(
            batch_size, layer.nofm, layer.hofm, layer.wofm, pidx)
        fp2c_dst.add(frng, coord + offset_dst)

    # Filters are read-only and known beforehand, can be easily replicated
    # in all memory nodes. Read from the nearest one.
    fil_coords_src = []
    dim_nodes_src = part_src.dim()
    for h, w in itertools.product(range(dim_nodes_src.h),
                                  range(dim_nodes_src.w)):
        fil_coords_src.append(PhyDim2(h, w) + offset_src)

    for pidx in part.gen_pidx():
        coord = part.coordinate(pidx)
        frng = part.part_fmap_range(
            batch_size, layer.nofm, layer.hofm, layer.wofm, pidx)

        frng_src = frng.corresponding_input_fmap_range(layer)

        ## ifmap access.

        coord_src_counts = fp2c_src.rget_counter(frng_src)
        assert sum(coord_src_counts.values()) == frng_src.size()
        for coord_src, cnt in coord_src_counts.items():
            nhops[de.IFM] += cnt * coord.hop_dist(coord_src)

        ## ofmap access.

        coord_dst_counts = fp2c_dst.rget_counter(frng)
        assert sum(coord_dst_counts.values()) == frng.size()
        for coord_dst, cnt in coord_dst_counts.items():
            nhops[de.OFM] += cnt * coord.hop_dist(coord_dst)

        ## filter access.

        fil_size = frng.size('n') * frng_src.size('n') * layer.filter_size()
        min_hops = min(coord.hop_dist(cfil) for cfil in fil_coords_src)
        nhops[de.FIL] += fil_size * min_hops

    return nhops

