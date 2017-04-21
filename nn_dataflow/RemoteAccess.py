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
from .FmapRange import FmapRangeMap
from .PhyDim2 import PhyDim2

'''
Parallel process partitioning among multiple 2D nodes.
'''

def get_nhops(coord1, coord2=PhyDim2(0, 0)):
    ''' Get number of hops from `coord1` to `coord2`. '''
    return abs(coord1.h - coord2.h) + abs(coord1.w - coord2.w)


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
            nhops[de.IFM] += cnt * get_nhops(coord_src, coord)

        ## ofmap access.

        coord_dst_counts = fp2c_dst.rget_counter(frng)
        assert sum(coord_dst_counts.values()) == frng.size()
        for coord_dst, cnt in coord_dst_counts.items():
            nhops[de.OFM] += cnt * get_nhops(coord_dst, coord)

        ## filter access.

        fil_size = frng.size('n') * frng_src.size('n') * layer.filter_size()
        min_hops = min(get_nhops(cfil, coord) for cfil in fil_coords_src)
        nhops[de.FIL] += fil_size * min_hops

    return nhops

