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

from nn_dataflow.core import partition
from nn_dataflow.core import ConvLayer
from nn_dataflow.core import DataCategoryEnum as de
from nn_dataflow.core import DataLayout
from nn_dataflow.core import FmapRange
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import PhyDim2

from . import TestPartitionFixture

class TestUnitNhopsToProcRegion(TestPartitionFixture):
    ''' Tests for unit_nhops_to_proc_region function. '''

    def test_small(self):
        ''' Small case with hand calculation. '''
        layer = ConvLayer(6, 8, 16, 3)
        assert self.batch_size == 8

        # i (0, 0), (2, 0): (0, 0, 0, 0) -- (4, 6, 10, 10)
        #   (0, 1), (2, 1): (0, 0, 0, 8) -- (4, 6, 10, 18)
        #   (0, 2), (2, 2): (4, 0, 0, 0) -- (8, 6, 10, 10)
        #   (0, 3), (2, 3): (4, 0, 0, 8) -- (8, 6, 10, 18)
        #   (1, 0), (3, 0): (0, 0, 8, 0) -- (4, 6, 18, 10)
        #   (1, 1), (3, 1): (0, 0, 8, 8) -- (4, 6, 18, 18)
        #   (1, 2), (3, 2): (4, 0, 8, 0) -- (8, 6, 18, 10)
        #   (1, 3), (3, 3): (4, 0, 8, 8) -- (8, 6, 18, 18)
        # o (0, 0): (0, 0, 0, 0) -- (4, 4, 8, 8)
        #   (0, 1): (0, 0, 0, 8) -- (4, 4, 8, 16)
        #   (0, 2): (4, 0, 0, 0) -- (8, 4, 8, 8)
        #   (0, 3): (4, 0, 0, 8) -- (8, 4, 8, 16)
        #   (1, 0): (0, 0, 8, 0) -- (4, 4, 16, 8)
        #   (1, 1): (0, 0, 8, 8) -- (4, 4, 16, 16)
        #   (1, 2): (4, 0, 8, 0) -- (8, 4, 16, 8)
        #   (1, 3): (4, 0, 8, 8) -- (8, 4, 16, 16)
        #   (2, 0): (0, 4, 0, 0) -- (4, 8, 8, 8)
        #   (2, 1): (0, 4, 0, 8) -- (4, 8, 8, 16)
        #   (2, 2): (4, 4, 0, 0) -- (8, 8, 8, 8)
        #   (2, 3): (4, 4, 0, 8) -- (8, 8, 8, 16)
        #   (3, 0): (0, 4, 8, 0) -- (4, 8, 16, 8)
        #   (3, 1): (0, 4, 8, 8) -- (4, 8, 16, 16)
        #   (3, 2): (4, 4, 8, 0) -- (8, 8, 16, 8)
        #   (3, 3): (4, 4, 8, 8) -- (8, 8, 16, 16)
        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 1), (2, 2), (1, 2), (1, 1)))

        nr = NodeRegion(origin=PhyDim2(0, 0), dim=part.dim(),
                        type=NodeRegion.PROC)

        # (0, 0, 0, 0) -- (4, 6, 18, 9): (-2, -2)
        # (0, 0, 0, 9) -- (4, 6, 18, 18): (-2, -1)
        # (4, 0, 0, 0) -- (8, 6, 18, 9): (-1, -2)
        # (4, 0, 0, 9) -- (8, 6, 18, 18): (-1, -1)
        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-2, -2),
            (2, 1), (1, 1), PhyDim2(2, 2))

        # (0, 0, 0, 0) -- (8, 4, 16, 8): (2, 2)
        # (0, 0, 0, 8) -- (8, 4, 16, 16): (2, 3)
        # (0, 4, 0, 0) -- (8, 8, 16, 8): (3, 2)
        # (0, 4, 0, 8) -- (8, 8, 16, 16): (3, 3)
        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(2, 2),
            (1, 1), (2, 1), PhyDim2(2, 2))

        filter_nodes = frozenset([PhyDim2(0, 0)])

        # filter: (0, 0) -> all, 6 * 4 * 3 * 3

        # ifmap: (-2, -2) -> (0, 0), (2, 0): 4 * 6 * 10 * 9
        #                 -> (0, 1), (2, 1): 4 * 6 * 10 * (9 - 8)
        #                 -> (1, 0), (3, 0): 4 * 6 * (18 - 8) * 9
        #                 -> (1, 1), (3, 1): 4 * 6 * (18 - 8) * (9 - 8)
        #        (-2, -1) -> (0, 0), (2, 0): 4 * 6 * 10 * (10 - 9)
        #                 -> (0, 1), (2, 1): 4 * 6 * 10 * (18 - 9)
        #                 -> (1, 0), (3, 0): 4 * 6 * (18 - 8) * (10 - 9)
        #                 -> (1, 1), (3, 1): 4 * 6 * (18 - 8) * (18 - 9)
        #        (-1, -2) -> (0, 2), (2, 2): (8 - 4) * 6 * 10 * 9
        #                 -> (0, 3), (2, 3): (8 - 4) * 6 * 10 * (9 - 8)
        #                 -> (1, 2), (3, 2): (8 - 4) * 6 * (18 - 8) * 9
        #                 -> (1, 3), (3, 3): (8 - 4) * 6 * (18 - 8) * (9 - 8)
        #        (-1, -1) -> (0, 2), (2, 2): (8 - 4) * 6 * 10 * (10 - 9)
        #                 -> (0, 3), (2, 3): (8 - 4) * 6 * 10 * (18 - 9)
        #                 -> (1, 2), (3, 2): (8 - 4) * 6 * (18 - 8) * (10 - 9)
        #                 -> (1, 3), (3, 3): (8 - 4) * 6 * (18 - 8) * (18 - 9)

        # ofmap: (2, 2) -> (0, 0):
        #               -> (0, 2):
        #               -> (1, 0):
        #               -> (1, 2): 4 * 4 * 8 * 8
        #        (2, 3) -> (0/1, 1/3)
        #        (3, 2) -> (2/3, 0/2)
        #        (3, 3) -> (2/3, 1/3)

        nhops = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        self.assertEqual(nhops[de.FIL],
                         6 * 4 * 3 * 3
                         * sum(h + w for h in range(4) for w in range(4)))
        self.assertEqual(nhops[de.IFM],
                         4 * 6 * 10 * ((4 + 6) * 9 + (5 + 7) * 1
                                       + (5 + 7) * 9 + (6 + 8) * 1
                                       + (3 + 5) * 1 + (4 + 6) * 9
                                       + (4 + 6) * 1 + (5 + 7) * 9
                                       + (5 + 7) * 9 + (6 + 8) * 1
                                       + (6 + 8) * 9 + (7 + 9) * 1
                                       + (4 + 6) * 1 + (5 + 7) * 9
                                       + (5 + 7) * 1 + (6 + 8) * 9))
        self.assertEqual(nhops[de.OFM],
                         4 * 4 * 8 * 8 * ((4 + 2 + 3 + 1) + (4 + 2 + 3 + 1)
                                          + (3 + 1 + 2 + 0) + (3 + 1 + 2 + 0)))

    def test_conv_layer(self):
        ''' CONV layers. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 1), (2, 4), (1, 2), (2, 1)))

        nr = NodeRegion(origin=PhyDim2(0, 0), dim=part.dim(),
                        type=NodeRegion.PROC)

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-3, -3),
            (1, 2), (4, 1), PhyDim2(8, 4))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(5, 5),
            (1, 1), (1, 2), PhyDim2(2, 4))

        filter_nodes = frozenset([PhyDim2(0, 0), PhyDim2(7, 7)])

        nhops = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        true_nhops = self._true_unit_nhops(
            layer, nr, part, filter_nodes, ilayout, olayout)

        self.assertListEqual(nhops, true_nhops)

    def test_fc_layer(self):
        ''' FC layers. '''
        layer = self.layers['FC']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((8, 1), (1, 1), (1, 2), (1, 4)))

        nr = NodeRegion(origin=PhyDim2(0, 0), dim=part.dim(),
                        type=NodeRegion.PROC)

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-3, 10),
            (1, 2), (4, 1), PhyDim2(4, 4))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(1, 1),
            (2, 1), (1, 2), PhyDim2(2, 2))

        filter_nodes = frozenset([PhyDim2(0, 0), PhyDim2(0, 7)])

        nhops = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        true_nhops = self._true_unit_nhops(
            layer, nr, part, filter_nodes, ilayout, olayout)

        self.assertListEqual(nhops, true_nhops)

    def test_pool_layer(self):
        ''' Pooling layers. '''
        layer = self.layers['POOL']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 1), (2, 4), (2, 2), (1, 1)))

        nr = NodeRegion(origin=PhyDim2(0, 0), dim=part.dim(),
                        type=NodeRegion.PROC)

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-3, -3),
            (1, 2), (4, 1), PhyDim2(8, 4))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(5, 5),
            (1, 2), (1, 1), PhyDim2(2, 4))

        filter_nodes = frozenset()

        nhops = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        true_nhops = self._true_unit_nhops(
            layer, nr, part, filter_nodes, ilayout, olayout)

        self.assertListEqual(nhops, true_nhops)

    def test_lr_layer(self):
        ''' LR layers. '''
        layer = self.layers['LR']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 1), (2, 4), (2, 2), (1, 1)))

        nr = NodeRegion(origin=PhyDim2(0, 0), dim=part.dim(),
                        type=NodeRegion.PROC)

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-3, -3),
            (1, 2), (4, 1), PhyDim2(8, 4))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(5, 5),
            (1, 1), (1, 2), PhyDim2(2, 4))

        filter_nodes = frozenset()

        nhops = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        true_nhops = self._true_unit_nhops(
            layer, nr, part, filter_nodes, ilayout, olayout)

        self.assertListEqual(nhops, true_nhops)

    def test_origin(self):
        ''' Origin. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((1, 1), (1, 1), (1, 1), (1, 1)))

        nr = NodeRegion(origin=PhyDim2(3, 3), dim=part.dim(),
                        type=NodeRegion.PROC)

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-3, -3),
            (1, 1), (1, 1), PhyDim2(1, 1))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(3, 3),
            (1, 1), (1, 1), PhyDim2(1, 1))

        filter_nodes = frozenset([PhyDim2(3, -3)])

        nhops_1 = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        nr = NodeRegion(origin=PhyDim2(6, 6), dim=part.dim(),
                        type=NodeRegion.PROC)

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-6, -6),
            (1, 1), (1, 1), PhyDim2(1, 1))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(6, 6),
            (1, 1), (1, 1), PhyDim2(1, 1))

        filter_nodes = frozenset([PhyDim2(6, -6)])

        nhops_2 = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        self.assertListEqual(nhops_2, [n * 2 for n in nhops_1])

    def test_ofmap_local(self):
        ''' With locally stored ofmaps. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((4, 1), (1, 1), (1, 4), (1, 1)))

        nr = NodeRegion(origin=PhyDim2(3, 3), dim=part.dim(),
                        type=NodeRegion.PROC)

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-3, -3),
            (1, 1), (1, 1), PhyDim2(1, 1))

        olayout = DataLayout(
            frngs=(FmapRange((0,) * 4,
                             (self.batch_size, layer.nofm,
                              layer.hofm, layer.wofm)),),
            regions=(nr,),
            parts=(part,))

        filter_nodes = frozenset([PhyDim2(3, -3)])

        nhops = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        self.assertEqual(nhops[de.OFM], 0)

    def test_use_fwd(self):
        ''' Use access forwarding. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 1), (2, 4), (1, 2), (2, 1)))

        nr = NodeRegion(origin=PhyDim2(0, 0), dim=part.dim(),
                        type=NodeRegion.PROC)

        far_dist = 1000

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-far_dist, 0),
            (1, 1), (1, 1), PhyDim2(1, 1))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(0, -far_dist),
            (1, 1), (1, 1), PhyDim2(1, 1))

        filter_nodes = frozenset([PhyDim2(far_dist, 0), PhyDim2(0, far_dist)])

        nhops_base = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['BASE'])
        nhops_accfwd = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['ACCFWD'])
        nhops_bufshr = partition.unit_nhops_to_proc_region(
            layer, self.batch_size, nr, part,
            filter_nodes, ilayout, olayout, self.options['BUFSHR'])

        for dce in range(de.NUM):
            self.assertEqual(nhops_accfwd[dce], nhops_bufshr[dce])

        # In the basic access scheme, FIL and IFM are independently fetched,
        # resulting in repeating remote fetch. OFM are merged locally and only
        # stored back remotely once.
        self.assertGreater(nhops_base[de.FIL],
                           layer.total_filter_size() * far_dist
                           * part.size(pe.BATP) * part.size(pe.OFMP) * 0.8)
        self.assertGreater(nhops_base[de.IFM],
                           layer.total_ifmap_size(self.batch_size) * far_dist
                           * part.size(pe.OUTP) * 0.8)

        p_layer, p_batch_size, _ = part.part_layer(layer, self.batch_size)
        # With forwarding, everyone is only remotely fetched once.
        self.assertLess(nhops_accfwd[de.FIL],
                        p_layer.total_filter_size()
                        * part.size(pe.INPP, pe.OUTP)
                        * (far_dist + nr.dim.size()))
        self.assertLess(nhops_accfwd[de.IFM],
                        p_layer.total_ifmap_size(p_batch_size)
                        * part.size(pe.INPP, pe.OFMP, pe.BATP)
                        * (far_dist + nr.dim.size()))
        self.assertLess(nhops_accfwd[de.OFM],
                        p_layer.total_ofmap_size(p_batch_size)
                        * part.size(pe.OUTP, pe.OFMP, pe.BATP)
                        * (far_dist + nr.dim.size()))

    def _make_data_layout(self, nfm, hfm, wfm, origin, bdim, ndim, dims):
        ''' Make a DataLayout instance. '''
        frng = FmapRange((0,) * 4, (self.batch_size, nfm, hfm, wfm))

        region = NodeRegion(origin=origin, dim=dims, type=NodeRegion.DRAM)

        # From top to bottom: h, w, b, n.
        order = (pe.OFMP, pe.BATP, pe.OUTP, pe.INPP)
        pdims = [None] * pe.NUM

        pdims[pe.BATP] = PhyDim2(*bdim)
        pdims[pe.OUTP] = PhyDim2(*ndim)
        pdims[pe.OFMP] = PhyDim2(h=dims.h // bdim[0] // ndim[0],
                                 w=dims.w // bdim[1] // ndim[1])
        pdims[pe.INPP] = PhyDim2(1, 1)

        part = PartitionScheme(order=order, pdims=pdims)

        return DataLayout(frngs=(frng,), regions=(region,), parts=(part,))

    def _true_unit_nhops(self, layer, region, part, filnodes, ilayout, olayout):
        '''
        Calculate the unit number of hops for i/ofmaps.
        '''
        nhops = [0] * de.NUM

        ifrmap = ilayout.fmap_range_map()
        ofrmap = olayout.fmap_range_map()

        for pidx in part.gen_pidx():

            # Current node.
            coord = part.coordinate(region, pidx)

            # Middle node of INPP.
            midpidx = list(pidx)
            midpidx[pe.INPP] = divmod(part.size(pe.INPP) // 2,
                                      part.dim(pe.INPP).w)
            midcoord = part.coordinate(region, midpidx)

            fil, ifr, ofr = partition.proc_data_range(
                layer, self.batch_size, part, pidx)

            # Ifmaps.
            for src, size in ifrmap.rget_counter(ifr).items():
                nhops[de.IFM] += coord.hop_dist(src) * size

            # Ofmaps.
            if coord == midcoord:
                # From/to sources.
                for src, size in ofrmap.rget_counter(ofr).items():
                    nhops[de.OFM] += coord.hop_dist(src) * size
            else:
                # From/to middle node.
                nhops[de.OFM] += ofr.size() * coord.hop_dist(midcoord) / 2

            # Filters.
            if filnodes:
                num_fils = fil[0].size() * fil[1].size()
                nhops[de.FIL] += num_fils * layer.filter_size() \
                        * min(coord.hop_dist(n) for n in filnodes)

        return nhops

