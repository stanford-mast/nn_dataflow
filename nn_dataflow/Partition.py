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

from . import ParallelEnum as pe
from . import Util
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

