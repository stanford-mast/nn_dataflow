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
from collections import namedtuple

from . import parallel_enum as pe
from .. import util
from .layer import ConvLayer, LocalRegionLayer
from .phy_dim2 import PhyDim2

PARTITION_SCHEME_LIST = ['order',
                         'pdims',
                        ]

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

    def part_layer(self, layer, batch_size):
        '''
        Get the partitioned layer structure and batch size. Return partitioned
        layer, partitioned batch size, and partitioning op occupation.
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

