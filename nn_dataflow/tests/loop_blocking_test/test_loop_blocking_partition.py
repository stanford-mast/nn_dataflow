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

from nn_dataflow.core import BufShrScheme
from nn_dataflow.core import DataCategoryEnum as de
from nn_dataflow.core import loop_blocking
from nn_dataflow.core import LoopBlockingScheme
from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow import util

from . import TestLoopBlockingFixture

class TestLoopBlockingPartition(TestLoopBlockingFixture):
    ''' Tests for LoopBlocking module with partitioning. '''

    def setUp(self):

        super(TestLoopBlockingPartition, self).setUp()

        # LoopBlockingScheme records stats of all nodes.
        self.total_ops = self.layer['PAR'].total_ops(self.batch_size)

        self.par_proc_region = self.resource['PAR'].proc_region

    def test_accfwd(self):
        ''' Scheme using accfwd. '''

        for part in self._gen_all_partition():

            p_nld = self._part_nld(part)

            filter_size, ifmap_size, ofmap_size = self._total_part_size(part)

            bufshr = BufShrScheme(self.par_proc_region, part)

            # Filter may still have redundant fetch.
            fil_fetch = part.size(pe.BATP, pe.OFMP) // bufshr.size(de.FIL)

            for lbs in loop_blocking.gen_loopblocking(
                    p_nld, self.resource['PAR'], part, self.none_cstr,
                    self.cost, self.options['ACCFWD']):
                if not lbs.is_valid():
                    continue

                # Ops.
                self.assertAlmostEqual(lbs.ops, self.total_ops)

                # Access forwarding reduction.
                accfwd_red = lbs.accfwd_reduction
                self.assertEqual(accfwd_red[de.FIL],
                                 part.size(pe.BATP, pe.OFMP) // fil_fetch)
                self.assertEqual(accfwd_red[de.OFM], part.size(pe.INPP))
                self.assertEqual(accfwd_red[de.IFM], part.size(pe.OUTP))

                # Top fetch and access.
                top_fetch = lbs.fetch[0]
                top_access = lbs.access[0]
                self.assertAlmostEqual(top_access[de.FIL],
                                       top_fetch[de.FIL] * filter_size
                                       * fil_fetch)
                self.assertAlmostEqual(top_access[de.OFM],
                                       top_fetch[de.OFM] * ofmap_size)
                self.assertGreaterEqual(top_access[de.IFM],
                                        top_fetch[de.IFM] * ifmap_size)

    def test_bufshr(self):
        ''' Scheme using bufshr. '''

        for part in self._gen_all_partition():

            p_nld = self._part_nld(part)

            bufshr = BufShrScheme(self.par_proc_region, part)

            # Filter may still have redundant fetch.
            fil_fetch = part.size(pe.BATP, pe.OFMP) // bufshr.size(de.FIL)

            for optkey in ['BUFSHR', 'BUFSHR-BYP']:

                for lbs in loop_blocking.gen_loopblocking(
                        p_nld, self.resource['PAR'], part, self.none_cstr,
                        self.cost, self.options[optkey]):
                    if not lbs.is_valid():
                        continue

                    # Ops.
                    self.assertAlmostEqual(lbs.ops, self.total_ops)

                    # Buffer sharing uses access forwarding reduction.
                    accfwd_red = lbs.accfwd_reduction
                    self.assertEqual(accfwd_red[de.FIL],
                                     part.size(pe.BATP, pe.OFMP) // fil_fetch)
                    self.assertEqual(accfwd_red[de.OFM], part.size(pe.INPP))
                    self.assertEqual(accfwd_red[de.IFM], part.size(pe.OUTP))

                    # Buffer sharing group size.
                    bufshr_grp_size = lbs.bufshr_grp_size
                    self.assertSequenceEqual(bufshr_grp_size, accfwd_red)

                    # Buffer sharing subgroup size.
                    bufshr_subgrp_size = lbs.bufshr_subgrp_size
                    self.assertTrue(all(subgrp <= grp for subgrp, grp
                                        in zip(bufshr_subgrp_size,
                                               bufshr_grp_size)))

    def test_bufshr_access(self):
        ''' Access of scheme using bufshr. '''

        for part in self._gen_all_partition():

            p_nld = self._part_nld(part)

            bufshr = BufShrScheme(self.par_proc_region, part)

            for lbs in loop_blocking.gen_loopblocking(
                    p_nld, self.resource['PAR'], part, self.none_cstr,
                    self.cost, self.options['BUFSHR']):
                if not lbs.is_valid():
                    continue

                # Skip those without bufshr.
                if all(sgs <= 1 for sgs in lbs.bufshr_subgrp_size):
                    continue

                # Sim.
                dram_access, gbuf_access, bufshr_stats = \
                        self._sim_access_conv(lbs, get_bufshr=True)

                self._verify_bufshr_stats(dram_access, gbuf_access,
                                          bufshr_stats, lbs, bufshr,
                                          'test_bufshr_access')

    def test_bufshr_access_byp(self):
        ''' Access of scheme using bufshr with bypassing. '''

        for part in self._gen_all_partition():

            p_nld = self._part_nld(part)

            bufshr = BufShrScheme(self.par_proc_region, part)

            for lbs in loop_blocking.gen_loopblocking(
                    p_nld, self.resource['PAR'], part, self.none_cstr,
                    self.cost, self.options['BUFSHR-BYP']):
                if not lbs.is_valid():
                    continue

                # Skip those without bufshr.
                if all(sgs <= 1 for sgs in lbs.bufshr_subgrp_size):
                    continue
                # Skip those without bypassing.
                if all(lbs.stored_in_gbuf):
                    continue

                # Sim.
                dram_access, gbuf_access, bufshr_stats = \
                        self._sim_access_conv(lbs, get_bufshr=True)

                self._verify_bufshr_stats(dram_access, gbuf_access,
                                          bufshr_stats, lbs, bufshr,
                                          'test_bufshr_access')

    def test_bufshr_rotation_example(self):
        ''' Example scheme using bufshr with rotation. '''

        # Make a PartitionScheme that allows bufshr for all data categories.
        part = PartitionScheme(order=range(pe.NUM),
                               pdims=((2, 1), (1, 2), (1, 1), (2, 1)))
        bufshr = BufShrScheme(self.par_proc_region, part)
        self.assertTrue(all(bufshr.size(dce) > 1 for dce in range(de.NUM)),
                        'test_bufshr_rotation_example: '
                        'made-up PartitionScheme is not expected: '
                        '{}, bufshr size {}'
                        .format(part,
                                [bufshr.size(dce) for dce in range(de.NUM)]))

        # Make a LoopBlockingScheme that uses bufshr for all data categories.
        p_nld = self._part_nld(part)
        bl_ts = ((util.idivc(p_nld.loopcnt[le.IFM], 6),
                  util.idivc(p_nld.loopcnt[le.OFM], 9),
                  util.idivc(p_nld.loopcnt[le.BAT], 2)),
                 (3, 3, 2),
                 (2, 3, 1))
        bl_ords = (tuple(range(le.NUM)), tuple(range(le.NUM)))
        lbs = LoopBlockingScheme(p_nld, bl_ts, bl_ords, self.resource['PAR'],
                                 bufshr, self.options['BUFSHR'])
        self.assertTrue(lbs.is_valid())
        self.assertGreater(sum(lbs.get_noc_access()), 0)
        self.assertTrue(all(sgs > 1 for sgs in lbs.bufshr_subgrp_size)
                        and all(t > 1 for t in bl_ts[0]),
                        'test_bufshr_rotation_example: '
                        'made-up LoopBlockingScheme is not expected: '
                        '{}, top factors {}, bufshr subgrp size {}'
                        .format((bl_ts, bl_ords), bl_ts[0],
                                lbs.bufshr_subgrp_size))

        # Sim.
        dram_access, gbuf_access, bufshr_stats = \
                self._sim_access_conv(lbs, get_bufshr=True)

        self._verify_bufshr_stats(dram_access, gbuf_access, bufshr_stats,
                                  lbs, bufshr, 'test_bufshr_rotation_example')

    def test_bufshr_skip_rot_example(self):
        ''' Example scheme using bufshr that skips the single rotation. '''

        # Make a PartitionScheme that allows bufshr for IFM.
        part = PartitionScheme(order=range(pe.NUM),
                               pdims=((2, 2), (1, 1), (2, 1), (1, 1)))
        bufshr = BufShrScheme(self.par_proc_region, part)
        self.assertEqual(bufshr.size(de.IFM), 4,
                         'test_bufshr_skip_rot_example: '
                         'made-up PartitionScheme is not expected: '
                         '{}, bufshr size for {} {}.'
                         .format(part, de.IFM, bufshr.size(de.IFM)))

        # Make a LoopBlockingScheme that has a single rotation for IFM.
        p_nld = self._part_nld(part)
        bl_ts = ((util.idivc(p_nld.loopcnt[le.IFM], 3),
                  util.idivc(p_nld.loopcnt[le.OFM], 3),
                  util.idivc(p_nld.loopcnt[le.BAT], 2)),
                 (1, 1, 2),
                 (3, 3, 1))
        bl_ords = (tuple(range(le.NUM)), tuple(range(le.NUM)))
        lbs = LoopBlockingScheme(p_nld, bl_ts, bl_ords, self.resource['PAR'],
                                 bufshr, self.options['BUFSHR'])
        self.assertTrue(lbs.is_valid())
        self.assertGreater(sum(lbs.get_noc_access()), 0)
        self.assertEqual(lbs.bufshr_subgrp_size[de.IFM], 4,
                         'test_bufshr_skip_rot_example: '
                         'made-up LoopBlockingScheme is not expected: '
                         '{}, bufshr subgrp size for {} {}.'
                         .format((bl_ts, bl_ords), de.IFM,
                                 lbs.bufshr_subgrp_size[de.IFM]))
        self.assertGreater(lbs.bufshr_wide_fetch_width[de.IFM], 1,
                           'test_bufshr_skip_rot_example: '
                           'made-up LoopBlockingScheme is not expected: '
                           '{}, bufshr wide fetch width for {} {}.'
                           .format((bl_ts, bl_ords), de.IFM,
                                   lbs.bufshr_wide_fetch_width[de.IFM]))
        self.assertEqual(lbs.bufshr_rot_round_cnt[de.IFM], 0,
                         'test_bufshr_skip_rot_example: '
                         'made-up LoopBlockingScheme is not expected: '
                         '{}, bufshr rotation rounds for {} {}'
                         .format((bl_ts, bl_ords), de.IFM,
                                 lbs.bufshr_rot_round_cnt[de.IFM]))

        # Sim.
        dram_access, gbuf_access, bufshr_stats = \
                self._sim_access_conv(lbs, get_bufshr=True)

        self._verify_bufshr_stats(dram_access, gbuf_access, bufshr_stats,
                                  lbs, bufshr,
                                  'test_bufshr_skip_rot_example')

    def test_bufshr_wide_fetch_example(self):
        ''' Example scheme using bufshr with wide fetch. '''

        # Make a PartitionScheme that allows bufshr for IFM.
        part = PartitionScheme(order=range(pe.NUM),
                               pdims=((2, 2), (1, 1), (2, 1), (1, 1)))
        bufshr = BufShrScheme(self.par_proc_region, part)
        self.assertEqual(bufshr.size(de.IFM), 4,
                         'test_bufshr_wide_fetch_example: '
                         'made-up PartitionScheme is not expected: '
                         '{}, bufshr size for {} {}.'
                         .format(part, de.IFM, bufshr.size(de.IFM)))

        for t1, t2 in [((3, 3, 1), (1, 1, 2)),
                       ((1, 3, 2), (3, 1, 1))]:
            # Make a LoopBlockingScheme that has wide fetch for IFM.
            p_nld = self._part_nld(part)
            bl_ts = (tuple(util.idivc(p_nld.loopcnt[lpe], t1[lpe] * t2[lpe])
                           for lpe in range(le.NUM)),
                     t1, t2)
            # At GBUF level, from inner to outer: le.BAT, le.IFM, le.OFM.
            bl_ords = (tuple(range(le.NUM)), (1, 2, 0))
            lbs = LoopBlockingScheme(p_nld, bl_ts, bl_ords,
                                     self.resource['PAR'], bufshr,
                                     self.options['BUFSHR'])
            self.assertTrue(lbs.is_valid())
            self.assertGreater(sum(lbs.get_noc_access()), 0)
            self.assertEqual(lbs.bufshr_subgrp_size[de.IFM], 4,
                             'test_bufshr_wide_fetch_example: '
                             'made-up LoopBlockingScheme is not expected: '
                             '{}, bufshr subgrp size for {} {}.'
                             .format((bl_ts, bl_ords), de.IFM,
                                     lbs.bufshr_subgrp_size[de.IFM]))
            self.assertGreater(lbs.bufshr_wide_fetch_width[de.IFM], 1,
                               'test_bufshr_wide_fetch_example: '
                               'made-up LoopBlockingScheme is not expected: '
                               '{}, bufshr wide fetch width for {} {}.'
                               .format((bl_ts, bl_ords), de.IFM,
                                       lbs.bufshr_wide_fetch_width[de.IFM]))
            self.assertGreater(lbs.bufshr_rot_round_cnt[de.IFM], 0,
                               'test_bufshr_wide_fetch_example: '
                               'made-up LoopBlockingScheme is not expected: '
                               '{}, bufshr rotation rounds for {} {}'
                               .format((bl_ts, bl_ords), de.IFM,
                                       lbs.bufshr_rot_round_cnt[de.IFM]))

            # Sim.
            dram_access, gbuf_access, bufshr_stats = \
                    self._sim_access_conv(lbs, get_bufshr=True)

            self._verify_bufshr_stats(dram_access, gbuf_access, bufshr_stats,
                                      lbs, bufshr,
                                      'test_bufshr_wide_fetch_example')

    def test_bufshr_multisubgrp_example(self):
        ''' Example scheme using bufshr with multiple subgroups in a group. '''

        # Make a PartitionScheme that allows bufshr for IFM.
        part = PartitionScheme(order=list(reversed(range(pe.NUM))),
                               pdims=((2, 2), (1, 1), (2, 1), (1, 1)))
        bufshr = BufShrScheme(self.par_proc_region, part)
        self.assertEqual(bufshr.size(de.IFM), 4,
                         'test_bufshr_multisubgrp_example: '
                         'made-up PartitionScheme is not expected: '
                         '{}, bufshr size for {} {}.'
                         .format(part, de.IFM, bufshr.size(de.IFM)))

        # Make a LoopBlockingScheme that has multi subgroups per group for IFM.
        p_nld = self._part_nld(part)
        bl_ts = ((util.idivc(p_nld.loopcnt[le.IFM], 1),
                  util.idivc(p_nld.loopcnt[le.OFM], 3),
                  util.idivc(p_nld.loopcnt[le.BAT], 2)),
                 (1, 3, 2),
                 (1, 1, 1))
        # At GBUF level, from inner to outer: le.BAT, le.OFM, le.IFM.
        bl_ords = (tuple(range(le.NUM)), (2, 1, 0))
        lbs = LoopBlockingScheme(p_nld, bl_ts, bl_ords, self.resource['PAR'],
                                 bufshr, self.options['BUFSHR'])
        self.assertTrue(lbs.is_valid())
        self.assertGreater(sum(lbs.get_noc_access()), 0)
        self.assertGreater(lbs.bufshr_grp_size[de.IFM],
                           lbs.bufshr_subgrp_size[de.IFM],
                           'test_bufshr_multisubgrp_example: '
                           'made-up LoopBlockingScheme is not expected: '
                           '{}, bufshr grp size {}, bufshr subgrp size {}'
                           .format((bl_ts, bl_ords), lbs.bufshr_grp_size,
                                   lbs.bufshr_subgrp_size))
        self.assertGreater(lbs.bufshr_rot_round_cnt[de.IFM], 0,
                           'test_bufshr_multisubgrp_example: '
                           'made-up LoopBlockingScheme is not expected: '
                           '{}, bufshr rotation rounds for {} {}'
                           .format((bl_ts, bl_ords), de.IFM,
                                   lbs.bufshr_rot_round_cnt[de.IFM]))

        # Sim.
        dram_access, gbuf_access, bufshr_stats = \
                self._sim_access_conv(lbs, get_bufshr=True)

        self._verify_bufshr_stats(dram_access, gbuf_access, bufshr_stats,
                                  lbs, bufshr,
                                  'test_bufshr_multisubgrp_example')

    def test_bufshr_get_noc_access(self):
        ''' get_noc_access of scheme using bufshr. '''

        for part in self._gen_all_partition():

            p_nld = self._part_nld(part)

            for lbs in loop_blocking.gen_loopblocking(
                    p_nld, self.resource['PAR'], part, self.none_cstr,
                    self.cost, self.options['BUFSHR']):

                noc_access = lbs.get_noc_access()

                if not lbs.is_valid():
                    self.assertIsNone(noc_access)

                else:
                    for dce in range(de.NUM):
                        self.assertAlmostEqual(
                            lbs.bufshr_rotation_access[dce]
                            + lbs.bufshr_wide_fetch_access[dce],
                            noc_access[dce])

    def test_bufshr_localregionlayer(self):
        ''' Scheme using bufshr for LocalRegionLayer. '''

        for part in self._gen_all_partition(layerkey='POOL'):

            p_nld = self._part_nld(part, layerkey='POOL')

            for lbs in loop_blocking.gen_loopblocking(
                    p_nld, self.resource['PAR'], part, self.none_cstr,
                    self.cost, self.options['BUFSHR']):
                if not lbs.is_valid():
                    continue

                self.assertTrue(all(gs == 1 for gs in lbs.bufshr_grp_size),
                                'test_bufshr_localregionlayer: '
                                'non-1 bufshr group size {}, part {}'
                                .format(lbs.bufshr_grp_size, part))

