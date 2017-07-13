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

from nn_dataflow import BufShrScheme
from nn_dataflow import DataCategoryEnum as de
from nn_dataflow import LoopBlocking
from nn_dataflow import ParallelEnum as pe

from . import TestLoopBlockingFixture

class TestLoopBlockingPartition(TestLoopBlockingFixture):
    ''' Tests for LoopBlocking module with partitioning. '''

    def setUp(self):

        super(TestLoopBlockingPartition, self).setUp()

        # LoopBlockingScheme records stats of all nodes.
        self.total_ops = self.layer['PAR'].total_ops(self.batch_size)

    def test_scheme_dict(self):
        ''' get_scheme_dict stats. '''

        for part, p_nld, p_occ in self._gen_all_partition():

            filter_size, ifmap_size, ofmap_size = self._total_part_size(part)

            fil_fetch = part.size(pe.BATP, pe.OFMP)
            ifm_fetch = part.size(pe.OUTP)
            ofm_fetch = part.size(pe.INPP)

            for lbs in LoopBlocking.gen_loopblocking(
                    p_nld, self.resource['PAR'], part, self.cost, p_occ,
                    self.options['BASE']):
                if not lbs.is_valid():
                    continue

                sch_dict = lbs.get_scheme_dict(self.cost)

                # Ops.
                self.assertAlmostEqual(sch_dict['ops'], self.total_ops)

                # Top fetch and access.
                top_fetch = sch_dict['fetch'][0]
                top_access = sch_dict['access'][0]
                self.assertAlmostEqual(top_access[de.FIL],
                                       top_fetch[de.FIL] * filter_size
                                       * fil_fetch)
                self.assertAlmostEqual(top_access[de.OFM],
                                       top_fetch[de.OFM] * ofmap_size
                                       * ofm_fetch)
                self.assertGreaterEqual(top_access[de.IFM],
                                        top_fetch[de.IFM] * ifmap_size
                                        * ifm_fetch)

    def test_accfwd(self):
        ''' scheme using accfwd. '''

        for part, p_nld, p_occ in self._gen_all_partition():

            filter_size, ifmap_size, ofmap_size = self._total_part_size(part)

            bufshr = BufShrScheme(part)

            # Filter may still have redundant fetch.
            fil_fetch = part.size(pe.BATP, pe.OFMP) // bufshr.size(de.FIL)

            for lbs in LoopBlocking.gen_loopblocking(
                    p_nld, self.resource['PAR'], part, self.cost, p_occ,
                    self.options['ACCFWD']):
                if not lbs.is_valid():
                    continue

                sch_dict = lbs.get_scheme_dict(self.cost)

                # Ops.
                self.assertAlmostEqual(sch_dict['ops'], self.total_ops)

                # Access forwarding reduction.
                accfwd_red = sch_dict['accfwd_reduction']
                self.assertEqual(accfwd_red[de.FIL],
                                 part.size(pe.BATP, pe.OFMP) // fil_fetch)
                self.assertEqual(accfwd_red[de.OFM], part.size(pe.INPP))
                self.assertEqual(accfwd_red[de.IFM], part.size(pe.OUTP))

                # Top fetch and access.
                top_fetch = sch_dict['fetch'][0]
                top_access = sch_dict['access'][0]
                self.assertAlmostEqual(top_access[de.FIL],
                                       top_fetch[de.FIL] * filter_size
                                       * fil_fetch)
                self.assertAlmostEqual(top_access[de.OFM],
                                       top_fetch[de.OFM] * ofmap_size)
                self.assertGreaterEqual(top_access[de.IFM],
                                        top_fetch[de.IFM] * ifmap_size)

    def test_bufshr(self):
        ''' scheme using bufshr. '''

        for part, p_nld, p_occ in self._gen_all_partition():

            bufshr = BufShrScheme(part)

            # Filter may still have redundant fetch.
            fil_fetch = part.size(pe.BATP, pe.OFMP) // bufshr.size(de.FIL)

            for optkey in ['BUFSHR', 'BUFSHR-BYP']:

                for lbs in LoopBlocking.gen_loopblocking(
                        p_nld, self.resource['PAR'], part, self.cost, p_occ,
                        self.options[optkey]):
                    if not lbs.is_valid():
                        continue

                    sch_dict = lbs.get_scheme_dict(self.cost)

                    # Ops.
                    self.assertAlmostEqual(sch_dict['ops'], self.total_ops)

                    # Buffer sharing uses access forwarding reduction.
                    accfwd_red = sch_dict['accfwd_reduction']
                    self.assertEqual(accfwd_red[de.FIL],
                                     part.size(pe.BATP, pe.OFMP) // fil_fetch)
                    self.assertEqual(accfwd_red[de.OFM], part.size(pe.INPP))
                    self.assertEqual(accfwd_red[de.IFM], part.size(pe.OUTP))

                    # Buffer sharing group size.
                    bufshr_grp_size = sch_dict['bufshr_grp_size']
                    self.assertSequenceEqual(bufshr_grp_size, accfwd_red)

                    # Buffer sharing subgroup size.
                    bufshr_subgrp_size = sch_dict['bufshr_subgrp_size']
                    self.assertTrue(all(subgrp <= grp for subgrp, grp
                                        in zip(bufshr_subgrp_size,
                                               bufshr_grp_size)))

