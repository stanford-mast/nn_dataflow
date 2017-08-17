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

from nn_dataflow import ConvLayer
from nn_dataflow import DataCategoryEnum as de
from nn_dataflow import DataLayout
from nn_dataflow import FmapRange, FmapRangeMap
from nn_dataflow import NodeRegion
from nn_dataflow import ParallelEnum as pe
from nn_dataflow import Partition
from nn_dataflow import PartitionScheme
from nn_dataflow import PhyDim2
from nn_dataflow import Util

from . import TestPartitionFixture

class TestPartLayerUnitNhops(TestPartitionFixture):
    ''' Tests for part_layer_unit_nhops function. '''

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

        # (0, 0, 0, 0): (0, 0, 0, 0) -- (4, 6, 18, 9): (-2, -2)
        # (0, 0, 0, 1): (0, 0, 0, 9) -- (4, 6, 18, 18): (-2, -1)
        # (1, 0, 0, 0): (4, 0, 0, 0) -- (8, 6, 18, 9): (-1, -2)
        # (1, 0, 0, 1): (4, 0, 0, 9) -- (8, 6, 18, 18): (-1, -1)
        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-2, -2),
            (2, 1, 1, 2), PhyDim2(2, 2))

        # (0, 0, 0, 0): (0, 0, 0, 0) -- (8, 4, 8, 16): (2, 2)
        # (0, 0, 1, 0): (0, 0, 8, 0) -- (8, 4, 16, 16): (2, 3)
        # (0, 1, 0, 0): (0, 4, 0, 0) -- (8, 8, 8, 16): (3, 2)
        # (0, 1, 1, 0): (0, 4, 8, 0) -- (8, 8, 16, 16): (3, 3)
        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(2, 2),
            (1, 2, 2, 1), PhyDim2(2, 2))

        filter_nodes = [PhyDim2(0, 0)]

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
        #               -> (0, 1):
        #               -> (0, 2):
        #               -> (0, 3): 4 * 4 * 8 * 8
        #        (2, 3) -> (1, 0/1/2/3)
        #        (3, 2) -> (2, 0/1/2/3)
        #        (3, 3) -> (3, 0/1/2/3)

        nhops = Partition.part_layer_unit_nhops(
            layer, self.batch_size, part, nr,
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
                         4 * 4 * 8 * 8 * ((4 + 3 + 2 + 3) + (4 + 3 + 2 + 1)
                                          + (3 + 2 + 1 + 2) + (3 + 2 + 1 + 0)))

    def test_conv_layer(self):
        ''' CONV layers. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 1), (2, 4), (1, 2), (2, 1)))

        nr = NodeRegion(origin=PhyDim2(0, 0), dim=part.dim(),
                        type=NodeRegion.PROC)

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-3, -3),
            (2, 4, 2, 2), PhyDim2(8, 4))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(5, 5),
            (1, 2, 2, 2), PhyDim2(2, 4))

        filter_nodes = [PhyDim2(0, 0), PhyDim2(7, 7)]

        nhops = Partition.part_layer_unit_nhops(
            layer, self.batch_size, part, nr,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        true_nhops = self._true_unit_nhops(
            layer, part, nr, filter_nodes, ilayout, olayout)

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
            (2, 4, 1, 2), PhyDim2(4, 4))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(1, 1),
            (2, 2, 1, 1), PhyDim2(2, 2))

        filter_nodes = [PhyDim2(0, 0), PhyDim2(0, 7)]

        nhops = Partition.part_layer_unit_nhops(
            layer, self.batch_size, part, nr,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        true_nhops = self._true_unit_nhops(
            layer, part, nr, filter_nodes, ilayout, olayout)

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
            (2, 4, 2, 2), PhyDim2(8, 4))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(5, 5),
            (1, 2, 2, 2), PhyDim2(2, 4))

        filter_nodes = []

        nhops = Partition.part_layer_unit_nhops(
            layer, self.batch_size, part, nr,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        true_nhops = self._true_unit_nhops(
            layer, part, nr, filter_nodes, ilayout, olayout)

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
            (2, 4, 2, 2), PhyDim2(8, 4))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(5, 5),
            (1, 2, 2, 2), PhyDim2(2, 4))

        filter_nodes = []

        nhops = Partition.part_layer_unit_nhops(
            layer, self.batch_size, part, nr,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        true_nhops = self._true_unit_nhops(
            layer, part, nr, filter_nodes, ilayout, olayout)

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
            (1, 1, 1, 1), PhyDim2(1, 1))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(3, 3),
            (1, 1, 1, 1), PhyDim2(1, 1))

        filter_nodes = [PhyDim2(3, -3)]

        nhops_1 = Partition.part_layer_unit_nhops(
            layer, self.batch_size, part, nr,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        nr = NodeRegion(origin=PhyDim2(6, 6), dim=part.dim(),
                        type=NodeRegion.PROC)

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-6, -6),
            (1, 1, 1, 1), PhyDim2(1, 1))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(6, 6),
            (1, 1, 1, 1), PhyDim2(1, 1))

        filter_nodes = [PhyDim2(6, -6)]

        nhops_2 = Partition.part_layer_unit_nhops(
            layer, self.batch_size, part, nr,
            filter_nodes, ilayout, olayout, self.options['BASE'])

        self.assertListEqual(nhops_2, [n * 2 for n in nhops_1])

    def test_use_accfwd(self):
        ''' Use access forwarding. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 1), (2, 4), (1, 2), (2, 1)))

        nr = NodeRegion(origin=PhyDim2(0, 0), dim=part.dim(),
                        type=NodeRegion.PROC)

        far_dist = 1000

        ilayout = self._make_data_layout(
            layer.nifm, layer.hifm, layer.wifm, PhyDim2(-far_dist, 0),
            (1, 1, 1, 1), PhyDim2(1, 1))

        olayout = self._make_data_layout(
            layer.nofm, layer.hofm, layer.wofm, PhyDim2(0, -far_dist),
            (1, 1, 1, 1), PhyDim2(1, 1))

        filter_nodes = [PhyDim2(far_dist, 0), PhyDim2(0, far_dist)]

        nhops_base = Partition.part_layer_unit_nhops(
            layer, self.batch_size, part, nr,
            filter_nodes, ilayout, olayout, self.options['BASE'])
        nhops_accfwd = Partition.part_layer_unit_nhops(
            layer, self.batch_size, part, nr,
            filter_nodes, ilayout, olayout, self.options['ACCFWD'])
        nhops_bufshr = Partition.part_layer_unit_nhops(
            layer, self.batch_size, part, nr,
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

    def _make_data_layout(self, nfm, hfm, wfm, origin, nums, dims):
        ''' Make a DataLayout instance. '''
        assert Util.prod(nums) == dims.size()

        def _coord(idxs):
            # In the order of n, b, w, h, i.e., 1, 0, 3, 2.
            cflat = 0
            for i in [1, 0, 3, 2]:
                cflat = cflat * nums[i] + idxs[i]
            assert cflat < dims.size()
            return PhyDim2(*divmod(cflat, dims.w))

        sizes = (self.batch_size, nfm, hfm, wfm)

        frmap = FmapRangeMap()

        for idxs in itertools.product(*[range(n) for n in nums]):

            begs = [i * s // n for i, n, s in zip(idxs, nums, sizes)]
            ends = [(i + 1) * s // n for i, n, s in zip(idxs, nums, sizes)]

            frmap.add(FmapRange(begs, ends), (_coord(idxs),))

        dl = DataLayout(frmap=frmap, origin=origin, type=NodeRegion.DATA)
        assert dl.frmap.complete_fmap_range().size() == Util.prod(sizes)

        return dl

    def _true_unit_nhops(self, layer, part, region, filnodes, ilayout, olayout):
        '''
        Calculate the unit number of hops for i/ofmaps.
        '''
        nhops = [0] * de.NUM

        for pidx in part.gen_pidx():

            coord = part.coordinate(region, pidx)
            # Middle node of INPP.
            midpidx = list(pidx)
            midpidx[pe.INPP] = divmod(part.size(pe.INPP) // 2,
                                      part.dim(pe.INPP).w)
            midcoord = part.coordinate(region, midpidx)

            # Ifmaps.
            ifr = Partition.part_layer_ifmap_range(
                layer, self.batch_size, part, pidx)
            for srcs, size in ilayout.frmap.rget_counter(ifr).items():
                nhops[de.IFM] += sum(coord.hop_dist(s + ilayout.origin)
                                     for s in srcs) * size

            # Ofmaps.
            ofr = Partition.part_layer_ofmap_range(
                layer, self.batch_size, part, pidx)
            if coord == midcoord:
                # From/to sources.
                for srcs, size in olayout.frmap.rget_counter(ofr).items():
                    nhops[de.OFM] += sum(coord.hop_dist(s + olayout.origin)
                                         for s in srcs) * size
            else:
                # From/to middle node.
                nhops[de.OFM] += ofr.size() * coord.hop_dist(midcoord) / 2

            # Filters.
            if filnodes:
                num_fils = ifr.size('n') * ofr.size('n')
                nhops[de.FIL] += num_fils * layer.filter_size() \
                        * min(coord.hop_dist(n) for n in filnodes)

        return nhops

