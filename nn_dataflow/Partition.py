'''
Parallel process partitioning.

Partition among multiple copies of the PE arrays.

For our case, only deal with up to 2D layout of PE arrays.
'''

import itertools
import numpy as np

from . import DataCategoryEnum as de
from . import ParallelEnum as pe
from . import Util
from .Layer import Layer
from .PhyDim2 import PhyDim2


class Partition2dScheme(object):
    '''
    Denote a 2D partitioning scheme.
    '''
    def __init__(self, order, partition2d):
        self.order = tuple(order)
        self.partition2d = tuple(partition2d)
        assert len(self.order) == pe.NUM and len(set(self.order)) == pe.NUM
        assert len(self.partition2d) == pe.NUM
        assert all([isinstance(p, PhyDim2) for p in self.partition2d])

    def total_size(self):
        '''
        Get total parallel partition size.
        '''
        return np.prod([p.h * p.w for p in self.partition2d])

    def gen_all_indexes2d(self):
        '''
        Generator to iterate over all 2D partition indexes.
        '''
        for indexes in itertools.product(
                *[itertools.product(*[range(dim) for dim in p])
                  for p in self.partition2d]):
            yield [PhyDim2._make(idx) for idx in indexes]

    def physical_coordinate2d(self, index2d):
        '''
        Get physical 2D coordinate from the given 2D partition index.
        '''
        coord = PhyDim2(0, 0)
        for penum in self.order:
            coord = [c * p + i for c, p, i
                     in zip(coord, self.partition2d[penum], index2d[penum])]
        return PhyDim2._make(coord)

    def as_pod_type(self):
        '''
        Return as a POD type (a tuple) to allow serialization/deserialization.
        '''
        return (self.order, self.partition2d)

    def __str__(self):
        return str(self.as_pod_type())


def gen_layer_partition2d(layer, dim_nodes):
    '''
    Iterate through all possible partitioning schemes that partition `layer`
    into 2D `dim_nodes` nodes.

    Return a tuple of the partition scheme and the partitioned layer
    parameters.
    '''
    for ph, pw in itertools.product(Util.factorize(dim_nodes.h, pe.NUM),
                                    Util.factorize(dim_nodes.w, pe.NUM)):
        # Require fmap partition the same factor in both dimension.
        if ph[pe.OFMP] != pw[pe.OFMP]:
            continue

        partition2d = [PhyDim2(h, w) for h, w in zip(ph, pw)]

        # Require partition is approximately dividable of total size.
        if not Util.approx_dividable(layer.nofm, partition2d[pe.OUTP].size()):
            continue
        if not Util.approx_dividable(layer.sofm, partition2d[pe.OFMP].h) \
            or not Util.approx_dividable(layer.sofm, partition2d[pe.OFMP].w):
            continue

        # Partitioned layer spec.
        layer_part = Layer(layer.nifm,
                           Util.idivc(layer.nofm, partition2d[pe.OUTP].size()),
                           Util.idivc(layer.sofm, partition2d[pe.OFMP].h),
                           layer.sfil,
                           layer.strd)

        # For different order.
        for order in itertools.permutations(tuple(range(pe.NUM))):
            # Size-(1, 1) partition has no effect, so its order is not
            # relevant. Force them at the beginning.
            no_partitions = [v for v in range(pe.NUM)
                             if partition2d[v].size() == 1]
            if not all([order[i] == no_partitions[i] for i
                        in range(len(no_partitions))]):
                continue

            yield (Partition2dScheme(order, partition2d), layer_part)


def gen_layer_naive_partition2d(layer, dim_nodes):
    '''
    Use naive way to partition `layer` into 2D `dim_nodes` nodes: for CONV
    layer use all OFMP partition; for FC layer use all OUTP partition.

    Return a tuple of the partition scheme and the partitioned layer
    parameters.
    '''
    if layer.sofm == 1:
        # FC layer: all OUTP.
        partition2d = [0] * pe.NUM
        partition2d[pe.OUTP] = dim_nodes
        partition2d[pe.OFMP] = PhyDim2(1, 1)
        # Force size-(1, 1) partition at the beginning.
        order = (pe.OFMP, pe.OUTP)
    else:
        # CONV layer: all OFMP.
        partition2d = [0] * pe.NUM
        partition2d[pe.OUTP] = PhyDim2(1, 1)
        partition2d[pe.OFMP] = dim_nodes
        # Force size-(1, 1) partition at the beginning.
        order = (pe.OUTP, pe.OFMP)

    layer_part = Layer(layer.nifm,
                       Util.idivc(layer.nofm, partition2d[pe.OUTP].size()),
                       Util.idivc(layer.sofm, partition2d[pe.OFMP].h),
                       layer.sfil,
                       layer.strd)

    yield (Partition2dScheme(order, partition2d), layer_part)


def get_layer_range(nfmap, sfmap, part, index):
    '''
    Get the range of the layer for the given partition index.

    `part` is the partition2d field of Partition2dScheme. `index` is a given
    partition2d index.
    '''
    # fmap channel partition.
    idx_chn = index[pe.OUTP].h * part[pe.OUTP].w + index[pe.OUTP].w
    n_beg = nfmap * idx_chn // part[pe.OUTP].size()
    n_end = nfmap * (idx_chn + 1) // part[pe.OUTP].size()
    assert n_end <= nfmap
    # fmap height tiling.
    h_beg = sfmap * index[pe.OFMP].h // part[pe.OFMP].h
    h_end = sfmap * (index[pe.OFMP].h + 1) // part[pe.OFMP].h
    assert h_end <= sfmap
    # fmap width tiling.
    w_beg = sfmap * index[pe.OFMP].w // part[pe.OFMP].w
    w_end = sfmap * (index[pe.OFMP].w + 1) // part[pe.OFMP].w
    assert w_end <= sfmap
    return (n_beg, n_end), (h_beg, h_end), (w_beg, w_end)


def unit_nhops_layer_partition2d(layer, batch_size, part_lcurr, part_lprev):
    '''
    Get total number of hops for each data category when partition the given
    layer with `part_lcurr` and 'part_lprev'.

    Return a tuple with each element being the number of hops for each data
    category.
    '''

    # Layers should have the same model parallel size (i.e., the same data
    # parallel size).
    assert part_lcurr.total_size() == part_lprev.total_size()

    # Per data category.
    nhops = [0 for _ in range(de.NUM)]

    if part_lcurr.total_size() == 1 and part_lprev.total_size() == 1:
        # Case with no partition.
        return nhops

    # Record all dest coordinates that need this element.
    req_lists_lprev = [[[[] for _ in range(layer.sifm)]
                        for _ in range(layer.sifm)]
                       for _ in range(layer.nifm)]

    for index_lcurr in part_lcurr.gen_all_indexes2d():
        # Physical coordinate.
        coord_lcurr = part_lcurr.physical_coordinate2d(index_lcurr)

        # Range of current layer for this index.
        _, h_lcurr_rng, w_lcurr_rng = get_layer_range(
            layer.nofm, layer.sofm, part_lcurr.partition2d, index_lcurr)

        # Range of previous layer, i.e., input for this pidx
        # ifmap channels. All.
        n_lprev_beg = 0
        n_lprev_end = layer.nifm
        # ifmap height tiling.
        # xy_i = xy_o * stride + (0 ... sfil-1)
        h_lprev_beg = h_lcurr_rng[0] * layer.strd
        h_lprev_end = (h_lcurr_rng[1] - 1) * layer.strd + layer.sfil
        assert h_lprev_end <= layer.sifm
        # ifmap width tiling.
        w_lprev_beg = w_lcurr_rng[0] * layer.strd
        w_lprev_end = (w_lcurr_rng[1] - 1) * layer.strd + layer.sfil
        assert w_lprev_end <= layer.sifm

        for n_lprev, h_lprev, w_lprev in itertools.product(
                range(n_lprev_beg, n_lprev_end),
                range(h_lprev_beg, h_lprev_end),
                range(w_lprev_beg, w_lprev_end)):
            req_lists_lprev[n_lprev][h_lprev][w_lprev].append(coord_lcurr)

    def get_nhops(coord1, coord2):
        ''' Get number of hops from `coord1` to `coord2`. '''
        return abs(coord1.h - coord2.h) + abs(coord1.w - coord2.w)

    # Sum up nhops based on recorded dest coordinates.
    for index_lprev in part_lprev.gen_all_indexes2d():
        # Physical coordinate.
        coord_lprev = part_lprev.physical_coordinate2d(index_lprev)

        # Range of previous layer for this index.
        n_lprev_rng, h_lprev_rng, w_lprev_rng = get_layer_range(
            layer.nifm, layer.sifm, part_lprev.partition2d, index_lprev)

        for n_lprev, h_lprev, w_lprev in itertools.product(
                range(*n_lprev_rng), range(*h_lprev_rng), range(*w_lprev_rng)):
            rlist = req_lists_lprev[n_lprev][h_lprev][w_lprev]
            nhops[de.IFM] += np.sum([get_nhops(coord_lprev, c) for c in rlist])

    nhops = [n * batch_size for n in nhops]

    return nhops

