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
import unittest

from nn_dataflow.core import partition
from nn_dataflow.core import ConvLayer, FCLayer, LocalRegionLayer, PoolingLayer
from nn_dataflow.core import Option
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PhyDim2
from nn_dataflow import util

class TestPartitionFixture(unittest.TestCase):
    ''' Base fixture class for Partition tests. '''

    def setUp(self):

        self.layers = {}
        self.layers['BASE'] = ConvLayer(64, 64, 28, 3)
        self.layers['FC'] = FCLayer(4096, 1000, 6)
        self.layers['POOL'] = PoolingLayer(32, 7, 3, strd=2)
        self.layers['LR'] = LocalRegionLayer(32, 7, nreg=5, sreg=1)
        # With irregular nifm/nofm.
        self.layers['IRR'] = ConvLayer(255, 383, 13, 3)
        # With small numbers of fmaps.
        self.layers['SM'] = ConvLayer(5, 3, 13, 3)
        # Super small networks. No partitioning schemes.
        self.layers['SSM1'] = ConvLayer(1, 1, 2, 3)
        self.layers['SSM2'] = FCLayer(2, 2)
        self.layers['SSM3'] = PoolingLayer(1, 2, 2)

        self.batch_size = 8

        self.dim_nodes = {}
        self.dim_nodes['BASE'] = PhyDim2(4, 4)
        self.dim_nodes['LG'] = PhyDim2(10, 10)
        self.dim_nodes['PRIME'] = PhyDim2(3, 3)

        self.options = {}
        # Irrelevant options.
        optdict = {'ntops': 10000}
        self.options['BASE'] = Option(partition_hybrid=True,
                                      partition_batch=True,
                                      partition_ifmaps=True,
                                      **optdict)
        self.options['NOBATP'] = Option(partition_hybrid=True,
                                        partition_batch=False,
                                        partition_ifmaps=True,
                                        **optdict)
        self.options['NOINPP'] = Option(partition_hybrid=True,
                                        partition_batch=True,
                                        partition_ifmaps=False,
                                        **optdict)
        self.options['NOHYB'] = Option(partition_hybrid=False,
                                       partition_batch=True,
                                       partition_ifmaps=False,
                                       **optdict)
        self.options['ACCFWD'] = Option(partition_hybrid=True,
                                        partition_batch=True,
                                        partition_ifmaps=True,
                                        hw_access_forwarding=True,
                                        **optdict)
        self.options['BUFSHR'] = Option(partition_hybrid=True,
                                        partition_batch=True,
                                        partition_ifmaps=True,
                                        hw_gbuf_sharing=True,
                                        **optdict)

    def _gen_partition(self, wlkey='BASE', dnkey='BASE', optkey='BASE',
                       guaranteed=False):
        ''' Generate PartitionScheme. '''
        for part in partition.gen_partition(self.layers[wlkey],
                                            self.batch_size,
                                            self.dim_nodes[dnkey],
                                            self.options[optkey],
                                            guaranteed=guaranteed):
            yield part

    def _gen_partition_full(self, wlkey='BASE', dnkey='BASE'):
        ''' Generate all PartitionScheme regardless of equivalence. '''

        layer = self.layers[wlkey]
        dim_nodes = self.dim_nodes[dnkey]

        for ph, pw in itertools.product(util.factorize(dim_nodes.h, pe.NUM),
                                        util.factorize(dim_nodes.w, pe.NUM)):

            pdims = [PhyDim2(h, w) for h, w in zip(ph, pw)]

            # BATP.
            if self.batch_size % pdims[pe.BATP].size() != 0:
                continue

            # OUTP.
            if not util.approx_dividable(layer.nofm, pdims[pe.OUTP].size()):
                continue

            # OFMP.
            if not util.approx_dividable(layer.hofm, pdims[pe.OFMP].h) \
                    or not util.approx_dividable(layer.wofm, pdims[pe.OFMP].w):
                continue

            # INPP.
            if isinstance(layer, ConvLayer):
                if not util.approx_dividable(layer.nifm,
                                             pdims[pe.INPP].size()):
                    continue
            elif isinstance(layer, LocalRegionLayer):
                if pdims[pe.INPP].size() > 1:
                    continue

            # Fully utilize one dimension.
            pdims_no_ofmp = pdims[:pe.OFMP] + pdims[pe.OFMP + 1:]
            if any(pd.h != 1 and pd.h != dim_nodes.h
                   and pd.w != 1 and pd.w != dim_nodes.w
                   for pd in pdims_no_ofmp):
                continue

            for order in itertools.permutations(range(pe.NUM)):

                # Batch parallelism should be at the top.
                filtered_order = [pae for pae in order
                                  if pdims[pae].size() > 1]
                if pe.BATP in filtered_order and filtered_order[0] != pe.BATP:
                    continue

                yield PartitionScheme(order=order, pdims=pdims)

