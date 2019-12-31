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
import math

from nn_dataflow.core import Cost
from nn_dataflow.core import DataCategoryEnum as de
from nn_dataflow.core import LoopBlockingScheme
from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import MemHierEnum as me

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
        lbs = self._lbs(self._make_bl_ts(
            (self.nld['BASE'].loopcnt[le.IFM], 1, 1),
            (self.nld['BASE'].loopcnt[le.OFM], 1, 1),
            (self.nld['BASE'].loopcnt[le.BAT], 1, 1)))
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
        self.assertTrue(math.isinf(sum([sum(a) for a in lbs.get_access()])))

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

    def test_access_cost(self):
        ''' get_access_cost. '''

        for bl_ts, bl_ords in self._gen_loopblocking_all():

            lbs = self._lbs(bl_ts, bl_ords)

            if not lbs.is_valid():
                continue

            access = [sum(a) for a in lbs.get_access()]

            cost = lbs.get_access_cost(self.cost)
            self.assertAlmostEqual(
                cost,
                + sum(a * c for a, c in zip(access, self.cost.mem_hier)))

    def test_access_cost_same_lbs(self):
        ''' get_access_cost same lbs. '''
        lbs = self._lbs(self._make_bl_ts((0, 1, 1), (1, 0, 1), (1, 1, 0)),
                        rsrckey='LG')
        self.assertTrue(lbs.is_valid())
        c1 = lbs.get_access_cost(Cost(mac_op=1, mem_hier=(200, 6, 2, 1),
                                      noc_hop=50, idl_unit=50))
        c2 = lbs.get_access_cost(Cost(mac_op=-1, mem_hier=(-200, -6, -2, -1),
                                      noc_hop=-50, idl_unit=-50))
        self.assertAlmostEqual(c1, -c2)

    def test_access_cost_invalid(self):
        ''' get_access_cost invalid. '''
        lbs = self._lbs(self._make_bl_ts((0, 1, 1), (0, 1, 1), (1, 1, 0)),
                        rsrckey='SM')
        self.assertFalse(lbs.is_valid())
        self.assertTrue(math.isinf(lbs.get_access_cost(self.cost)))

    def test_ordered_loops(self):
        ''' Get ordered_loops. '''
        assert list(range(le.NUM)) == [le.IFM, le.OFM, le.BAT]

        self.assertListEqual(
            LoopBlockingScheme.ordered_loops((3, 5, 2), (2, 0, 1)),
            [(le.IFM, 3), (le.BAT, 2), (le.OFM, 5)])

        # Trivial loops at different positions.
        self.assertListEqual(
            LoopBlockingScheme.ordered_loops((3, 5, 1), (0, 1, 2)),
            [(le.OFM, 5), (le.IFM, 3)])
        self.assertListEqual(
            LoopBlockingScheme.ordered_loops((3, 5, 1), (1, 2, 0)),
            [(le.OFM, 5), (le.IFM, 3)])
        self.assertListEqual(
            LoopBlockingScheme.ordered_loops((3, 5, 1), (0, 2, 1)),
            [(le.OFM, 5), (le.IFM, 3)])

        # Different loops are trivial.
        self.assertListEqual(
            LoopBlockingScheme.ordered_loops((1, 5, 2), (0, 2, 1)),
            [(le.OFM, 5), (le.BAT, 2)])
        self.assertListEqual(
            LoopBlockingScheme.ordered_loops((3, 1, 2), (0, 2, 1)),
            [(le.BAT, 2), (le.IFM, 3)])

        # Multiple trivial loops.
        self.assertListEqual(
            LoopBlockingScheme.ordered_loops((1, 5, 1), (0, 1, 2)),
            [(le.OFM, 5)])
        self.assertListEqual(
            LoopBlockingScheme.ordered_loops((1, 1, 1), (0, 1, 2)),
            [])

        for bl_t, bl_ord in itertools.product(
                itertools.product(*[range(1, 8)] * 3),
                itertools.permutations(range(le.NUM))):

            ord_loops = LoopBlockingScheme.ordered_loops(bl_t, bl_ord)
            self.assertTrue(all(len(tpl) == 2 for tpl in ord_loops))
            self.assertFalse(any(tpl[1] <= 1 for tpl in ord_loops))
            self.assertEqual(len(ord_loops), le.NUM - bl_t.count(1))
            self.assertTrue(all(tpl[1] == bl_t[tpl[0]] for tpl in ord_loops))

            rev_loops = LoopBlockingScheme.ordered_loops(bl_t, bl_ord,
                                                         reverse=True)
            ord_lpes = LoopBlockingScheme.ordered_loops(bl_t, bl_ord,
                                                        lpe_only=True)
            self.assertEqual(len(rev_loops), len(ord_loops))
            self.assertEqual(len(ord_lpes), len(ord_loops))
            self.assertListEqual(list(reversed(rev_loops)), ord_loops)
            self.assertListEqual([tpl[0] for tpl in ord_loops], ord_lpes)

    def test_data_region_fetch(self):
        ''' PROC type data regions. '''

        # Multiple fetches with normal DATA regions.
        bl_ts = self._make_bl_ts((0, 1, 1), (0, 1, 1), (0, 1, 1))
        bl_ords = [[0] * le.NUM for _ in range(2)]
        bl_ords[0][le.IFM] = 1
        bl_ords[0][le.OFM] = 2
        bl_ords[0][le.BAT] = 0
        bl_ords[1] = range(le.NUM)
        lbs_norm = self._lbs(bl_ts, bl_ords)
        self.assertTrue(lbs_norm.is_valid())
        self.assertGreater(lbs_norm.fetch[0][de.IFM], 1)
        self.assertGreater(lbs_norm.fetch[0][de.OFM], 1)

        lbs = self._lbs(bl_ts, bl_ords, rsrckey='SRCNOTDATA')
        self.assertFalse(lbs.is_valid())
        lbs = self._lbs(bl_ts, bl_ords, rsrckey='DSTNOTDATA')
        self.assertFalse(lbs.is_valid())

        # Single top-level fetch.
        bl_ts = self._make_bl_ts((1, 0, 1), (1, 0, 1), (1, 0, 1))
        lbs_norm = self._lbs(bl_ts, rsrckey='LG')

        lbs = self._lbs(bl_ts, rsrckey='SRCNOTDATA')
        self.assertTrue(lbs.is_valid())
        self.assertLess(lbs.get_access_cost(self.cost),
                        lbs_norm.get_access_cost(self.cost))
        self.assertAlmostEqual(lbs_norm.get_access_cost(self.cost)
                               - lbs.get_access_cost(self.cost),
                               lbs.remote_gbuf_access[de.IFM]
                               * (self.cost.mem_hier_at(me.DRAM)
                                  - self.cost.mem_hier_at(me.GBUF)))
        self.assertAlmostEqual(lbs.access[me.DRAM][de.FIL],
                               lbs_norm.access[me.DRAM][de.FIL])
        self.assertAlmostEqual(lbs.access[me.DRAM][de.IFM], 0)
        self.assertAlmostEqual(lbs.access[me.DRAM][de.OFM],
                               lbs_norm.access[me.DRAM][de.OFM])
        self.assertAlmostEqual(lbs.access[me.GBUF][de.IFM],
                               lbs_norm.access[me.GBUF][de.IFM])
        self.assertAlmostEqual(lbs.remote_gbuf_access[de.IFM],
                               lbs_norm.access[me.DRAM][de.IFM])

        lbs = self._lbs(bl_ts, bl_ords, rsrckey='DSTNOTDATA')
        self.assertTrue(lbs.is_valid())
        self.assertLess(lbs.get_access_cost(self.cost),
                        lbs_norm.get_access_cost(self.cost))
        self.assertAlmostEqual(lbs_norm.get_access_cost(self.cost)
                               - lbs.get_access_cost(self.cost),
                               lbs.remote_gbuf_access[de.OFM]
                               * (self.cost.mem_hier_at(me.DRAM)
                                  - self.cost.mem_hier_at(me.GBUF)))
        self.assertAlmostEqual(lbs.access[me.DRAM][de.FIL],
                               lbs_norm.access[me.DRAM][de.FIL])
        self.assertAlmostEqual(lbs.access[me.DRAM][de.IFM],
                               lbs_norm.access[me.DRAM][de.IFM])
        self.assertAlmostEqual(lbs.access[me.DRAM][de.OFM], 0)
        self.assertAlmostEqual(lbs.access[me.GBUF][de.OFM],
                               lbs_norm.access[me.GBUF][de.OFM])
        self.assertAlmostEqual(lbs.remote_gbuf_access[de.OFM],
                               lbs_norm.access[me.DRAM][de.OFM])

        lbs = self._lbs(bl_ts, bl_ords, rsrckey='DATALOCAL')
        self.assertTrue(lbs.is_valid())
        self.assertLess(lbs.get_access_cost(self.cost),
                        lbs_norm.get_access_cost(self.cost))
        self.assertAlmostEqual(lbs.access[me.DRAM][de.FIL],
                               lbs_norm.access[me.DRAM][de.FIL])
        self.assertAlmostEqual(lbs.access[me.DRAM][de.IFM], 0)
        self.assertAlmostEqual(lbs.access[me.DRAM][de.OFM], 0)
        self.assertAlmostEqual(lbs.access[me.GBUF][de.IFM],
                               lbs_norm.access[me.GBUF][de.IFM])
        self.assertAlmostEqual(lbs.access[me.GBUF][de.OFM],
                               lbs_norm.access[me.GBUF][de.OFM])
        self.assertAlmostEqual(lbs.remote_gbuf_access[de.IFM],
                               lbs_norm.access[me.DRAM][de.IFM])
        self.assertAlmostEqual(lbs.remote_gbuf_access[de.OFM],
                               lbs_norm.access[me.DRAM][de.OFM])

    def test_fil_pinning(self):
        ''' Filter pinning. '''

        bl_ts = self._make_bl_ts((1, 0, 1), (1, 0, 1), (0, 1, 1))
        bl_ords = [range(le.NUM) for _ in range(2)]

        lbs_norm = self._lbs(bl_ts, bl_ords)
        self.assertTrue(lbs_norm.is_valid())
        self.assertGreater(lbs_norm.fetch[0][de.FIL], 0)
        self.assertGreater(lbs_norm.get_access()[0][de.FIL], 0)

        lbs = self._lbs(bl_ts, bl_ords, rsrckey='FILPIN')
        self.assertTrue(lbs.is_valid())
        self.assertEqual(lbs.fetch[0][de.FIL], 0)
        self.assertEqual(lbs.get_access()[0][de.FIL], 0)

