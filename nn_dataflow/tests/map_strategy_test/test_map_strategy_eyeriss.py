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

from nn_dataflow.core import DataDimLoops
from nn_dataflow.core import DataCategoryEnum as de
from nn_dataflow.core import Layer, ConvLayer, LocalRegionLayer
from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import MapStrategyEyeriss
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow import util

from . import TestMapStrategyFixture

class TestMapStrategyEyeriss(TestMapStrategyFixture):
    ''' Tests for MapStrategyEyeriss class. '''

    def setUp(self):

        super(TestMapStrategyEyeriss, self).setUp()

        self.dim_array = self.resource['BASE'].dim_array

    def test_invalid_layer(self):
        ''' Constructor with invalid layer type. '''
        with self.assertRaisesRegex(TypeError, 'MapEyeriss: .*type.*'):
            _ = MapStrategyEyeriss(Layer(1, 1), 4, 1, self.dim_array)

    def test_nested_loop_desc_sanity(self):
        ''' Generated nested loop description sanity check. '''

        batch_size = 4
        occ = 1

        for layer in tuple(self.convlayers.values()) + \
                     tuple(self.fclayers.values()) + \
                     tuple(self.lrlayers.values()) + \
                     tuple(self.fake_layers.values()):

            ms = MapStrategyEyeriss(layer, batch_size, occ, self.dim_array)

            for nld in ms.gen_nested_loop_desc():

                # Replication reduces numbers of IFM/OFM.
                self.assertGreaterEqual(layer.nifm, nld.loopcnt[le.IFM])
                self.assertGreaterEqual(layer.nofm, nld.loopcnt[le.OFM])
                # Folding increases batch size.
                self.assertEqual(nld.loopcnt[le.BAT] % batch_size, 0)

                # Total and unit ops.
                self.assertAlmostEqual(nld.total_ops(),
                                       layer.total_ops(batch_size))
                self.assertAlmostEqual(nld.unit_ops * util.prod(nld.loopcnt),
                                       layer.total_ops(batch_size))

                # Unit time and unit ops.
                # The difference is due to the loop occupancy, which is not
                # counted in utilization.
                self.assertGreaterEqual(
                    nld.unit_time * ms.utilization() * self.dim_array.size(),
                    nld.unit_ops)

                # Total access at DRAM.
                self.assertAlmostEqual(nld.total_access_at_of(me.DRAM, de.FIL),
                                       layer.total_filter_size()
                                       if isinstance(layer, ConvLayer) else 0)
                # IFM may have refetch due to folding.
                self.assertGreaterEqual(nld.total_access_at_of(me.DRAM, de.IFM)
                                        + 1e-7,
                                        layer.total_ifmap_size(batch_size))
                self.assertAlmostEqual(nld.total_access_at_of(me.DRAM, de.OFM),
                                       layer.total_ofmap_size(batch_size))

                # Unit access to REGF.
                self.assertAlmostEqual(nld.unit_access[me.REGF][de.FIL]
                                       * util.prod(nld.loopcnt),
                                       layer.total_ops(batch_size)
                                       if isinstance(layer, ConvLayer) else 0)
                self.assertAlmostEqual(nld.unit_access[me.REGF][de.IFM]
                                       * util.prod(nld.loopcnt),
                                       layer.total_ops(batch_size))
                self.assertAlmostEqual(nld.unit_access[me.REGF][de.OFM]
                                       * util.prod(nld.loopcnt),
                                       layer.total_ops(batch_size))

                # Unit GBUF size and unit access to DRAM.
                self.assertTrue(all(us >= ua for us, ua
                                    in zip(nld.usize_gbuf,
                                           nld.unit_access[me.DRAM])))

                # Unit REGF size.
                if isinstance(layer, ConvLayer):
                    # See JSSC'17, IV. A. Dimensions Beyond 2-D in PE Array. 1).
                    self.assertEqual(nld.usize_regf[de.FIL], layer.wfil)
                    self.assertEqual(nld.usize_regf[de.IFM], layer.wfil)
                    self.assertEqual(nld.usize_regf[de.OFM], 1)

                # Data dimension loops.
                if isinstance(layer, ConvLayer):
                    self.assertEqual(nld.data_loops[de.FIL],
                                     DataDimLoops(le.IFM, le.OFM))
                    self.assertEqual(nld.data_loops[de.IFM],
                                     DataDimLoops(le.IFM, le.BAT))
                    self.assertEqual(nld.data_loops[de.OFM],
                                     DataDimLoops(le.OFM, le.BAT))
                elif isinstance(layer, LocalRegionLayer):
                    self.assertEqual(nld.data_loops[de.FIL],
                                     DataDimLoops())
                    self.assertEqual(nld.data_loops[de.IFM],
                                     DataDimLoops(le.OFM, le.BAT))
                    self.assertEqual(nld.data_loops[de.OFM],
                                     DataDimLoops(le.OFM, le.BAT))

    def test_nested_loop_desc_occupancy(self):
        ''' Nested loop description with occupancy. '''

        batch_size = 4
        occ0 = 1
        occ1 = 0.8

        for layer in tuple(self.convlayers.values()) + \
                     tuple(self.fclayers.values()) + \
                     tuple(self.lrlayers.values()) + \
                     tuple(self.fake_layers.values()):

            ms0 = MapStrategyEyeriss(layer, batch_size, occ0, self.dim_array)
            ms1 = MapStrategyEyeriss(layer, batch_size, occ1, self.dim_array)

            for nld0, nld1 in zip(ms0.gen_nested_loop_desc(),
                                  ms1.gen_nested_loop_desc()):

                self.assertEqual(nld0.unit_time, nld1.unit_time)

                self.assertTupleEqual(nld0.usize_gbuf, nld1.usize_gbuf)
                self.assertTupleEqual(nld0.usize_regf, nld1.usize_regf)

                self.assertAlmostEqual(nld0.unit_ops * occ1,
                                       nld1.unit_ops * occ0)

                for mhe in range(me.NUM):
                    for dce in range(de.NUM):
                        if mhe == me.REGF:
                            self.assertAlmostEqual(
                                nld0.unit_access_at_of(mhe, dce) * occ1,
                                nld1.unit_access_at_of(mhe, dce) * occ0)
                        else:
                            self.assertAlmostEqual(
                                nld0.unit_access_at_of(mhe, dce),
                                nld1.unit_access_at_of(mhe, dce))

    def test_nested_loop_desc_fold_w(self):
        ''' Generated nested loop description when folding width. '''

        layer = self.convlayers['conv1']
        batch_size = 4
        occ = 1

        ms = MapStrategyEyeriss(layer, batch_size, occ, self.dim_array)

        self.assertTupleEqual(ms.repl, (1, 1))
        self.assertEqual(ms.fold.h, 1)
        self.assertGreater(ms.fold.w, 1)

        # Only 1 possible nld.
        nld_list = list(ms.gen_nested_loop_desc())
        self.assertEqual(len(nld_list), 1)
        nld = nld_list[0]

        # Fold to batch size.
        fold_w = ms.fold.w
        folded_layer = ConvLayer(layer.nifm, layer.nofm,
                                 (util.idivc(layer.hofm, fold_w), layer.wofm),
                                 (layer.hfil, layer.wfil),
                                 strd=(layer.htrd, layer.wtrd))
        folded_batch_size = batch_size * fold_w

        locc = layer.total_ops(batch_size) \
                / folded_layer.total_ops(folded_batch_size)
        self.assertLessEqual(locc, 1)

        self.assertEqual(nld.loopcnt[le.IFM], folded_layer.nifm)
        self.assertEqual(nld.loopcnt[le.OFM], folded_layer.nofm)
        self.assertEqual(nld.loopcnt[le.BAT], folded_batch_size)

        self.assertEqual(nld.usize_gbuf[de.FIL], folded_layer.filter_size())
        self.assertEqual(nld.usize_gbuf[de.IFM], folded_layer.ifmap_size())
        self.assertEqual(nld.usize_gbuf[de.OFM], folded_layer.ofmap_size())

        # DRAM and GBUF accesses are equal.
        self.assertTupleEqual(nld.unit_access[me.DRAM],
                              nld.unit_access[me.GBUF])

    def test_nested_loop_desc_fold_h(self):
        ''' Generated nested loop description when folding height. '''

        layer = self.fake_layers['LGFIL']
        batch_size = 4
        occ = 1

        ms = MapStrategyEyeriss(layer, batch_size, occ, self.dim_array)

        self.assertTupleEqual(ms.repl, (1, 1))
        self.assertGreater(ms.fold.h, 1)
        self.assertEqual(ms.fold.w, 1)

        # Only 1 possible nld.
        nld_list = list(ms.gen_nested_loop_desc())
        self.assertEqual(len(nld_list), 1)
        nld = nld_list[0]

        # Fold within processing pass.
        fold_h = ms.fold.h

        self.assertEqual(nld.loopcnt[le.IFM], layer.nifm)
        self.assertEqual(nld.loopcnt[le.OFM], layer.nofm)
        self.assertEqual(nld.loopcnt[le.BAT], batch_size)

        self.assertEqual(nld.usize_gbuf[de.FIL], layer.filter_size())
        self.assertEqual(nld.usize_gbuf[de.IFM], layer.ifmap_size())
        self.assertEqual(nld.usize_gbuf[de.OFM], layer.ofmap_size())

        # GBUF access is multiple of DRAM access.
        self.assertEqual(nld.unit_access_at_of(me.DRAM, de.FIL),
                         nld.unit_access_at_of(me.GBUF, de.FIL))
        self.assertEqual(nld.unit_access_at_of(me.DRAM, de.IFM) * fold_h,
                         nld.unit_access_at_of(me.GBUF, de.IFM))
        self.assertEqual(nld.unit_access_at_of(me.DRAM, de.OFM) * fold_h,
                         nld.unit_access_at_of(me.GBUF, de.OFM))

    def test_map_alex_net(self):
        ''' Map AlexNet, JSSC'17, Table III and V. '''

        # Replication is denoted in Table III as r and t. Physical PE set width
        # is denoted in Table III as e.
        # In Table III for CONV1, t = 2, but e = 7. Here we simplify to t = 1
        # and e = 14.
        repl_size_dict = {'conv1': 1 * 1,
                          'conv2': 1 * 1,
                          'conv3': 1 * 4,
                          'conv4': 2 * 2,
                          'conv5': 2 * 2}
        ppeset_width_dict = {'conv1': 14,
                             'conv2': 27,
                             'conv3': 13,
                             'conv4': 13,
                             'conv5': 13}

        # Active PEs given in Table V.
        active_pes_dict = {'conv1': 154,
                           'conv2': 135,
                           'conv3': 156,
                           'conv4': 156,
                           'conv5': 156}

        batch_size = 4
        occ = 1

        for name, layer in self.convlayers.items():

            ms = MapStrategyEyeriss(layer, batch_size, occ, self.dim_array)

            # Two ways to calculate active PEs.
            # Physical PE set size. Max active PEs.
            active_pes_max = ms.dim_ppeset.size()
            # Utilization. Average active PEs.
            active_pes_avg = ms.utilization() * self.dim_array.size()

            repl_size = ms.repl.size()

            # Note that the physical PE set width is given by flpeset, before
            # scheduling fold.w using repl.h.
            ppeset_width = ms.dim_flpeset.w

            self.assertTrue(active_pes_max == active_pes_dict[name]
                            or active_pes_avg == active_pes_dict[name])
            self.assertEqual(repl_size, repl_size_dict[name])
            self.assertEqual(ppeset_width, ppeset_width_dict[name])

