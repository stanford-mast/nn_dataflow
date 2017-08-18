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

import copy
import itertools
import math

from nn_dataflow.core import Cost
from nn_dataflow.core import DataCategoryEnum as de
from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow import util

from . import TestLoopBlockingFixture

class TestLoopBlockingScheme(TestLoopBlockingFixture):
    ''' Tests for LoopBlockingScheme. '''

    def test_is_valid(self):
        ''' Whether is_valid. '''
        # REGF size fails early.
        lbs = self._lbs(self._make_bl_ts((1, 1, 0), (0, 1, 1), (1, 1, 0)))
        self.assertFalse(lbs.is_valid())
        self.assertFalse(hasattr(lbs, 'fetch'))

        # GBUF size fails early.
        lbs = self._lbs(self._make_bl_ts((0, 1, 1), (1, 0, 1), (1, 0, 1)),
                        rsrckey='SM')
        self.assertFalse(lbs.is_valid())
        self.assertFalse(hasattr(lbs, 'fetch'))

        # GBUF size fails at recheck.
        lbs = self._lbs(self._make_bl_ts((0, 1, 1), (1, 0, 1), (1, 0, 1)),
                        rsrckey='SM', optkey='BYP')
        self.assertFalse(lbs.is_valid())
        self.assertTrue(hasattr(lbs, 'fetch'))

        # Valid.
        lbs = self._lbs(self._make_bl_ts((0, 1, 1), (0, 1, 1), (0, 1, 1)))
        self.assertTrue(lbs.is_valid())

    def test_data_size(self):
        ''' Get data_size. '''
        for si, so, sb in itertools.product([1, 2, 4], [1, 2, 4], [1, 2, 4]):

            # REGF size.
            lbs = self._lbs(self._make_bl_ts((0, 1, si), (0, 1, so),
                                             (0, 1, sb)), rsrckey='LG')
            self.assertTrue(lbs.is_valid())
            self.assertEqual(lbs.data_size(1, de.FIL),
                             si * so * self.nld['BASE'].usize_regf_of(de.FIL))
            self.assertEqual(lbs.data_size(1, de.IFM),
                             si * sb * self.nld['BASE'].usize_regf_of(de.IFM))
            self.assertEqual(lbs.data_size(1, de.OFM),
                             so * sb * self.nld['BASE'].usize_regf_of(de.OFM))
            self.assertEqual(lbs.data_size(1),
                             si * so * self.nld['BASE'].usize_regf_of(de.FIL)
                             + si * sb * self.nld['BASE'].usize_regf_of(de.IFM)
                             + so * sb * self.nld['BASE'].usize_regf_of(de.OFM))

            # GBUF size.
            lbs = self._lbs(self._make_bl_ts((0, si, 1), (0, so, 1),
                                             (0, sb, 1)), rsrckey='LG')
            self.assertTrue(lbs.is_valid())
            self.assertEqual(lbs.data_size(0, de.FIL),
                             si * so * self.nld['BASE'].usize_gbuf_of(de.FIL))
            self.assertEqual(lbs.data_size(0, de.IFM),
                             si * sb * self.nld['BASE'].usize_gbuf_of(de.IFM))
            self.assertEqual(lbs.data_size(0, de.OFM),
                             so * sb * self.nld['BASE'].usize_gbuf_of(de.OFM))
            self.assertEqual(lbs.data_size(0),
                             si * so * self.nld['BASE'].usize_gbuf_of(de.FIL)
                             + si * sb * self.nld['BASE'].usize_gbuf_of(de.IFM)
                             + so * sb * self.nld['BASE'].usize_gbuf_of(de.OFM))
            self.assertTrue(all(lbs.stored_in_gbuf))

    def test_data_size_bypass(self):
        ''' Get data_size bypass. '''
        for si, so, sb in itertools.product([1, 2, 4], [1, 2, 4], [1, 2, 4]):

            # GBUF size.
            lbs = self._lbs(self._make_bl_ts((0, si, 1), (0, so, 1),
                                             (0, sb, 1)), optkey='BYP')
            if lbs.is_valid():
                if not lbs.stored_in_gbuf[de.FIL]:
                    self.assertEqual(lbs.data_size(0, de.FIL), 0)
                if not lbs.stored_in_gbuf[de.IFM]:
                    self.assertEqual(lbs.data_size(0, de.IFM), 0)
                if not lbs.stored_in_gbuf[de.OFM]:
                    self.assertEqual(lbs.data_size(0, de.OFM), 0)
                self.assertEqual(lbs.data_size(0),
                                 lbs.data_size(0, de.FIL)
                                 + lbs.data_size(0, de.IFM)
                                 + lbs.data_size(0, de.OFM))

    def test_data_size_inv_args(self):
        ''' Get data_size invalid args. '''
        lbs = self._lbs(self._make_bl_ts((0, 1, 1), (0, 1, 1), (0, 1, 1)))

        with self.assertRaises(IndexError):
            _ = lbs.data_size(3)

        with self.assertRaises(IndexError):
            _ = lbs.data_size(0, 4)

    def test_access(self):
        ''' get_access. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all():

            lbs = self._lbs(bl_ts, bl_ords, rsrckey='LG')

            self.assertTrue(lbs.is_valid())
            self.assertSequenceEqual(bl_ts, lbs.bl_ts)
            self.assertSequenceEqual(bl_ords, lbs.bl_ords)

            # Model.
            access = lbs.get_access()
            # Sim.
            dram_access, gbuf_access = self._sim_access_conv(lbs)

            self.assertListEqual(access[me.DRAM], dram_access,
                                 'test_access: DRAM: '
                                 'model {} vs. sim {}. lbs: {} {}.'
                                 .format(access[me.DRAM], dram_access,
                                         bl_ts, bl_ords))
            self.assertListEqual(access[me.GBUF], gbuf_access,
                                 'test_access: GBUF: '
                                 'model {} vs. sim {}. lbs: {} {}.'
                                 .format(access[me.GBUF], gbuf_access,
                                         bl_ts, bl_ords))
            self.assertListEqual(access[me.REGF],
                                 [lbs.ops, lbs.ops, lbs.ops * 2])

    def test_access_bypass(self):
        ''' get_access bypass. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all():

            lbs = self._lbs(bl_ts, bl_ords, rsrckey='LG', optkey='BYP')

            self.assertTrue(lbs.is_valid())
            if all(lbs.stored_in_gbuf):
                continue

            # Model.
            access = lbs.get_access()
            # Sim.
            dram_access, gbuf_access = self._sim_access_conv(lbs)

            self.assertListEqual(access[me.DRAM], dram_access,
                                 'test_access_bypass: DRAM: '
                                 'model {} vs. sim {}. lbs: {} {}. '
                                 'stored in gbuf {}.'
                                 .format(access[me.DRAM], dram_access,
                                         bl_ts, bl_ords,
                                         lbs.stored_in_gbuf))
            self.assertListEqual(access[me.GBUF], gbuf_access,
                                 'test_access_bypass: GBUF: '
                                 'model {} vs. sim {}. lbs: {} {}. '
                                 'stored in gbuf {}.'
                                 .format(access[me.GBUF], gbuf_access,
                                         bl_ts, bl_ords,
                                         lbs.stored_in_gbuf))

    def test_access_bypass_lgfil(self):
        ''' get_access bypass for ConvLayer with large filter size. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all(wlkey='LGFIL'):

            lbs = self._lbs(bl_ts, bl_ords, wlkey='LGFIL', optkey='BYP')
            if not lbs.is_valid():
                continue
            if all(lbs.stored_in_gbuf):
                continue

            # Model.
            access = lbs.get_access()
            # Sim.
            dram_access, gbuf_access = self._sim_access_conv(lbs)

            self.assertListEqual(access[me.DRAM], dram_access,
                                 'test_access_bypass_lgfil: DRAM: '
                                 'model {} vs. sim {}. lbs: {} {}. '
                                 'stored in gbuf {}.'
                                 .format(access[me.DRAM], dram_access,
                                         bl_ts, bl_ords,
                                         lbs.stored_in_gbuf))
            self.assertListEqual(access[me.GBUF], gbuf_access,
                                 'test_access_bypass_lgfil: GBUF: '
                                 'model {} vs. sim {}. lbs: {} {}. '
                                 'stored in gbuf {}.'
                                 .format(access[me.GBUF], gbuf_access,
                                         bl_ts, bl_ords,
                                         lbs.stored_in_gbuf))

    def test_access_pool(self):
        ''' get_access for PoolingLayer. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all(wlkey='POOL'):

            lbs = self._lbs(bl_ts, bl_ords, wlkey='POOL', rsrckey='LG')

            self.assertTrue(lbs.is_valid())
            self.assertSequenceEqual(bl_ts, lbs.bl_ts)
            self.assertSequenceEqual(bl_ords, lbs.bl_ords)

            self.assertSequenceEqual(lbs.fetch[0], (1, 1, 1))
            self.assertSequenceEqual(lbs.fetch[1], (1, 1, 1))

    def test_access_invalid(self):
        ''' get_access invalid. '''
        lbs = self._lbs(self._make_bl_ts((0, 1, 1), (0, 1, 1), (1, 1, 0)),
                        rsrckey='SM')
        self.assertFalse(lbs.is_valid())
        self.assertIsNone(lbs.get_access())

    def test_top_level_fetch(self):
        ''' get_top_level_fetch. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all():

            lbs = self._lbs(bl_ts, bl_ords, rsrckey='LG')

            self.assertTrue(lbs.is_valid())

            # Top fetch.
            top_fetch = lbs.get_top_level_fetch()
            # Top access.
            top_access = lbs.get_access()[me.DRAM]

            self.assertEqual(top_access[de.FIL],
                             top_fetch[de.FIL]
                             * self.layer['BASE'].total_filter_size())
            self.assertEqual(top_access[de.IFM],
                             top_fetch[de.IFM]
                             * self.layer['BASE']
                             .total_ifmap_size(self.batch_size))
            self.assertEqual(top_access[de.OFM],
                             top_fetch[de.OFM]
                             * self.layer['BASE']
                             .total_ofmap_size(self.batch_size))

    def test_top_level_fetch_bypass(self):
        ''' get_top_level_fetch bypass. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all():

            lbs = self._lbs(bl_ts, bl_ords, rsrckey='LG', optkey='BYP')

            self.assertTrue(lbs.is_valid())
            if all(lbs.stored_in_gbuf):
                continue

            # Top fetch.
            top_fetch = lbs.get_top_level_fetch()
            # Top access.
            top_access = lbs.get_access()[me.DRAM]

            self.assertEqual(top_access[de.FIL],
                             top_fetch[de.FIL]
                             * self.layer['BASE'].total_filter_size())
            self.assertEqual(top_access[de.IFM],
                             top_fetch[de.IFM]
                             * self.layer['BASE']
                             .total_ifmap_size(self.batch_size))
            self.assertEqual(top_access[de.OFM],
                             top_fetch[de.OFM]
                             * self.layer['BASE']
                             .total_ofmap_size(self.batch_size))

    def test_top_level_fetch_invalid(self):
        ''' get_top_level_fetch invalid. '''
        lbs = self._lbs(self._make_bl_ts((0, 1, 1), (0, 1, 1), (1, 1, 0)),
                        rsrckey='SM')
        self.assertFalse(lbs.is_valid())
        self.assertIsNone(lbs.get_top_level_fetch())

    def test_cost(self):
        ''' get_cost. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all():

            lbs = self._lbs(bl_ts, bl_ords)

            if not lbs.is_valid():
                continue

            access = [sum(a) for a in lbs.get_access()]
            ops = lbs.ops
            nhops = 0
            time = lbs.time

            cost = lbs.get_cost(self.cost)
            self.assertAlmostEqual(
                cost,
                ops * self.cost.mac_op
                + sum(a * c for a, c in zip(access, self.cost.mem_hier))
                + nhops * self.cost.noc_hop
                + time * self.cost.unit_static)

    def test_cost_same_lbs(self):
        ''' get_cost same lbs. '''
        lbs = self._lbs(self._make_bl_ts((0, 1, 1), (1, 0, 1), (1, 1, 0)),
                        rsrckey='LG')
        self.assertTrue(lbs.is_valid())
        c1 = lbs.get_cost(Cost(mac_op=1, mem_hier=(200, 6, 2, 1),
                               noc_hop=50, unit_static=50))
        c2 = lbs.get_cost(Cost(mac_op=-1, mem_hier=(-200, -6, -2, -1),
                               noc_hop=-50, unit_static=-50))
        self.assertAlmostEqual(c1, -c2)

    def test_cost_invalid(self):
        ''' get_cost invalid. '''
        lbs = self._lbs(self._make_bl_ts((0, 1, 1), (0, 1, 1), (1, 1, 0)),
                        rsrckey='SM')
        self.assertFalse(lbs.is_valid())
        self.assertTrue(math.isinf(lbs.get_cost(self.cost)))

    def test_scheme_dict(self):
        ''' get_scheme_dict. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all():

            lbs = self._lbs(bl_ts, bl_ords, part_occ=self.part_occ)

            if not lbs.is_valid():
                self.assertIsNone(lbs.get_scheme_dict(self.cost))
                continue

            sdict = lbs.get_scheme_dict(self.cost)

            self.assertAlmostEqual(sdict['cost'], lbs.get_cost(self.cost))
            self.assertAlmostEqual(sdict['ops'], lbs.ops)
            self.assertAlmostEqual(sdict['time'], lbs.time)

            self.assertEqual(id(sdict['access']), id(lbs.get_access()))
            for lvl in [0, 1]:
                for dce in range(de.NUM):
                    self.assertAlmostEqual(sdict['size'][lvl][dce],
                                           lbs.data_size(lvl, dce))

            self.assertAlmostEqual(sdict['part_occ'], self.part_occ)

            self.assertEqual(util.prod(sdict['ti']),
                             self.nld['BASE'].loopcnt[le.IFM])
            self.assertEqual(util.prod(sdict['to']),
                             self.nld['BASE'].loopcnt[le.OFM])
            self.assertEqual(util.prod(sdict['tb']),
                             self.nld['BASE'].loopcnt[le.BAT])

    def test_scheme_dict_eval_order(self):
        ''' get_scheme_dict eval order. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all():

            lbs1 = self._lbs(bl_ts, bl_ords, rsrckey='LG')

            lbs2 = copy.deepcopy(lbs1)

            access1 = lbs1.get_access()
            sdict1 = lbs1.get_scheme_dict(self.cost)

            sdict2 = lbs2.get_scheme_dict(self.cost)
            access2 = lbs2.get_access()

            self.assertAlmostEqual(sdict1['cost'], sdict2['cost'])
            for mhe in range(me.NUM):
                for dce in range(de.NUM):
                    self.assertAlmostEqual(access1[mhe][dce],
                                           access2[mhe][dce])

    def test_part_occ(self):
        ''' Impact of part_occ. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all():

            lbs = self._lbs(bl_ts, bl_ords, part_occ=1, rsrckey='LG')

            lbs_ = copy.deepcopy(lbs)
            ops = lbs_.get_scheme_dict(self.cost)['ops']
            time = lbs_.get_scheme_dict(self.cost)['time']
            del lbs_

            for part_occ in [0.9, 0.8, 0.7]:
                lbs_ = copy.deepcopy(lbs)
                lbs_.part_occ = part_occ

                self.assertAlmostEqual(lbs_.get_scheme_dict(self.cost)['ops'],
                                       ops * part_occ)
                self.assertAlmostEqual(lbs_.get_scheme_dict(self.cost)['time'],
                                       time)

