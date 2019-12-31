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

import itertools
from collections import namedtuple

from . import parallel_enum as pe
from .. import util
from .fmap_range import FmapPosition, FmapRange
from .layer import ConvLayer, LocalRegionLayer
from .phy_dim2 import PhyDim2

PARTITION_SCHEME_LIST = ['order',
                         'pdims',
                        ]

PARA_ENUM_APPL2FRNG = [pe.BATP, pe.OUTP, pe.OFMP]

class PartitionScheme(namedtuple('PartitionScheme', PARTITION_SCHEME_LIST)):
    '''
    A parallel processing partitioning scheme.
    '''

    def __new__(cls, order, pdims):
        '''
        `order` is the order of partitioning hierarchy, from first to last
        is from top to bottom level on physical node array.

        `pdims` is a sequence of partitioning dimensions, each for one
        parallelism indexed by ParallelEnum.
        '''

        if len(order) != pe.NUM or set(order) != set(range(pe.NUM)):
            raise ValueError('PartitionScheme: order must be a permutation of '
                             '0 to {}.'.format(pe.NUM - 1))
        order_ = tuple(order)

        if len(pdims) != pe.NUM:
            raise ValueError('PartitionScheme: pdims must have length {}.'
                             .format(pe.NUM))
        try:
            pdims_ = tuple(PhyDim2(*dim) for dim in pdims)
        except TypeError:
            raise ValueError('PartitionScheme: elements in pdims must have '
                             'length 2.')

        ntp = super(PartitionScheme, cls).__new__(
            cls, order=order_, pdims=pdims_)

        return ntp

    def dim(self, *paes):
        '''
        Get the partitioning dimension for the given parallelisms. If not given,
        return total dimension.
        '''
        if not paes:
            return self.dim(*range(pe.NUM))

        dim = PhyDim2(1, 1)
        for pae in paes:
            dim *= self.pdims[pae]
        return dim

    def size(self, *paes):
        '''
        Get the partitioning size for the given parallelisms. If not given,
        return total size.
        '''
        return self.dim(*paes).size()

    def gen_pidx(self):
        '''
        Generator to iterate over all partition indexes.
        '''
        # Generator for all parallelisms.
        gens = []
        for dim in self.pdims:
            # This generator will go through all indexes for one parallelism.
            g = itertools.product(*[range(d) for d in dim])
            gens.append(g)

        for pidx in itertools.product(*gens):
            yield tuple(PhyDim2(*idx) for idx in pidx)

    def coordinate(self, node_region, pidx):
        '''
        Get the physical absolute 2D coordinate from the given partition index
        in the given node region.
        '''
        coord = [0, 0]
        for penum in self.order:
            coord = [c * d + i for c, d, i
                     in zip(coord, self.pdims[penum], pidx[penum])]
        return node_region.rel2abs(PhyDim2(*coord))

    def fmap_range(self, frng, pidx):
        '''
        Get the partitioned fmap range for the given partition index.
        '''
        fp_beg = frng.fp_beg
        fp_end = frng.fp_end

        # Batch partition.
        idx_bat = pidx[pe.BATP].h * self.pdims[pe.BATP].w + pidx[pe.BATP].w
        b_beg, b_end = util.get_ith_range((fp_beg.b, fp_end.b), idx_bat,
                                          self.pdims[pe.BATP].size())

        # Ofmap channel partition.
        idx_ofm = pidx[pe.OUTP].h * self.pdims[pe.OUTP].w + pidx[pe.OUTP].w
        n_beg, n_end = util.get_ith_range((fp_beg.n, fp_end.n), idx_ofm,
                                          self.pdims[pe.OUTP].size())

        # Fmap height tiling.
        h_beg, h_end = util.get_ith_range((fp_beg.h, fp_end.h), pidx[pe.OFMP].h,
                                          self.pdims[pe.OFMP].h)

        # Fmap width tiling.
        w_beg, w_end = util.get_ith_range((fp_beg.w, fp_end.w), pidx[pe.OFMP].w,
                                          self.pdims[pe.OFMP].w)

        return FmapRange(FmapPosition(b=b_beg, n=n_beg, h=h_beg, w=w_beg),
                         FmapPosition(b=b_end, n=n_end, h=h_end, w=w_end))

    def is_applicable_to_fmap_range(self):
        '''
        Whether this partitioning scheme is applicable to fmap ranges.
        '''
        return self.size() == self.size(*PARA_ENUM_APPL2FRNG)

    def part_layer(self, layer, batch_size):
        '''
        Get the partitioned layer structure and batch size. Return partitioned
        layer, partitioned batch size, and partitioning op occupancy.
        '''

        p_nifm = util.idivc(layer.nifm, self.pdims[pe.INPP].size())
        p_nofm = util.idivc(layer.nofm, self.pdims[pe.OUTP].size())
        p_hofm = util.idivc(layer.hofm, self.pdims[pe.OFMP].h)
        p_wofm = util.idivc(layer.wofm, self.pdims[pe.OFMP].w)

        if isinstance(layer, ConvLayer):
            p_layer = ConvLayer(p_nifm, p_nofm, (p_hofm, p_wofm),
                                (layer.hfil, layer.wfil),
                                strd=(layer.htrd, layer.wtrd))
        elif isinstance(layer, LocalRegionLayer):
            if self.pdims[pe.INPP].size() > 1:
                raise ValueError('PartitionScheme: input partitioning is '
                                 'invalid for LocalRegionLayer.')
            p_layer = LocalRegionLayer(p_nofm, (p_hofm, p_wofm),
                                       layer.nreg, (layer.hreg, layer.wreg),
                                       strd=(layer.htrd, layer.wtrd))
        else:
            raise TypeError('PartitionScheme: unrecognized layer type.')

        p_batch_size = util.idivc(batch_size, self.pdims[pe.BATP].size())

        p_occ = 1. * layer.total_ops(batch_size) \
                / (p_layer.total_ops(p_batch_size) * self.size())
        assert p_occ <= 1 + 1e-6

        return p_layer, p_batch_size, p_occ

    def part_neighbor_dist(self, node_region, pae):
        '''
        Get the 2D distance between nearest neighbor nodes with the given
        parallelism in the given node region.

        The returned neighbor distance is a PhyDim2 instance, each dimension of
        which is the hop distance to the neighbor on that logical dimension.
        '''
        if pae not in range(pe.NUM):
            return PhyDim2(float('nan'), float('nan'))

        hdist = []
        wdist = []

        for pidx in self.gen_pidx():
            coord = self.coordinate(node_region, pidx)
            # On logical h dimension.
            if pidx[pae].h > 0:
                pidx_ph = [pidx[p] - PhyDim2(h=1, w=0) if p == pae
                           else pidx[p] for p in range(pe.NUM)]
                coord_ph = self.coordinate(node_region, pidx_ph)
                hdist.append(coord.hop_dist(coord_ph))
            # On logical w dimension.
            if pidx[pae].w > 0:
                pidx_pw = [pidx[p] - PhyDim2(h=0, w=1) if p == pae
                           else pidx[p] for p in range(pe.NUM)]
                coord_pw = self.coordinate(node_region, pidx_pw)
                wdist.append(coord.hop_dist(coord_pw))

        # Average.
        hd = 1. * sum(hdist) / len(hdist) if hdist else float('inf')
        wd = 1. * sum(wdist) / len(wdist) if wdist else float('inf')

        return PhyDim2(h=hd, w=wd)

    def projection(self, region, appl2frng=False):
        '''
        Get the projection of the partitioning scheme onto a new NodeRegion
        `region`.

        If `appl2frng` is True, the projection must be applicable to fmap
        ranges.
        '''
        if region.dim.size() == 0:
            raise ValueError('PartitionScheme: '
                             'cannot project onto an empty node region.')

        order = self.order
        pdims = list(self.pdims)

        if appl2frng:
            # Shrink the partitioning not applicable to fmap ranges.
            for pae in range(pe.NUM):
                if pae not in PARA_ENUM_APPL2FRNG:
                    pdims[pae] = PhyDim2(1, 1)

        part_dim = util.prod(pdims)
        region_dim = region.dim

        # Keep the same order, and adjust each dimension.
        pdims = [list(pd) for pd in pdims]

        for di in range(2):
            pd = part_dim[di]
            rd = region_dim[di]

            if rd > pd:
                # New region dimension is larger. Extend.
                ext = rd // pd
                # Apply the extension to the top level.
                top_pae = PARA_ENUM_APPL2FRNG[0]
                pdims[top_pae][di] *= ext
            else:
                # Otherwise shrink.
                # Go from bottom to top. Keep bottom (first) levels unchanged,
                # and shrink top (latter) levels.
                for pae in reversed(order):
                    if rd >= pdims[pae][di]:
                        # Remaining size in region dimension is enough for
                        # current level. Use it to keep the current level the
                        # same size.
                        rd //= pdims[pae][di]
                    else:
                        # Remaining size in region dimension is not enough.
                        # Shrink current level to be whatever remains.
                        pdims[pae][di] = rd
                        rd = 1

        part = PartitionScheme(order=order, pdims=pdims)

        assert all(pd <= rd for pd, rd in zip(part.dim(), region_dim))
        assert not appl2frng or part.is_applicable_to_fmap_range()

        return part

