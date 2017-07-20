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

import unittest

from nn_dataflow import BufShrScheme
from nn_dataflow import DataCategoryEnum as de
from nn_dataflow import ParallelEnum as pe
from nn_dataflow import PartitionScheme

class TestBufShrScheme(unittest.TestCase):
    ''' Tests for BufShrScheme. '''

    def setUp(self):
        self.ps1 = PartitionScheme(order=[pe.BATP, pe.OUTP, pe.OFMP, pe.INPP],
                                   pdims=[(2, 3), (3, 1), (1, 5), (5, 2)])
        self.ps2 = PartitionScheme(order=range(pe.NUM),
                                   pdims=[(2, 2), (5, 5), (3, 3), (1, 1)])
        self.ps3 = PartitionScheme(order=range(pe.NUM),
                                   pdims=[(1, 6), (1, 2), (4, 1), (3, 5)])

        self.bufshr1 = BufShrScheme(self.ps1)
        self.bufshr2 = BufShrScheme(self.ps2)
        self.bufshr3 = BufShrScheme(self.ps3)

    def test_dim(self):
        ''' Accessor dim. '''
        for bufshr, ps in zip([self.bufshr1, self.bufshr2, self.bufshr3],
                              [self.ps1, self.ps2, self.ps3]):
            self.assertTupleEqual(bufshr.dim(de.IFM), ps.dim(pe.OUTP))
            self.assertTupleEqual(bufshr.dim(de.OFM), ps.dim(pe.INPP))

        self.assertTupleEqual(self.bufshr1.dim(de.FIL), self.ps1.dim(pe.OFMP))
        self.assertTupleEqual(self.bufshr2.dim(de.FIL),
                              self.ps2.dim(pe.OFMP, pe.BATP))
        self.assertTupleEqual(self.bufshr3.dim(de.FIL),
                              self.ps3.dim(pe.OFMP, pe.BATP))

    def test_size(self):
        ''' Get size. '''
        for bufshr in [self.bufshr1, self.bufshr2, self.bufshr3]:
            for dce in range(de.NUM):
                self.assertEqual(bufshr.dim(dce).size(), bufshr.size(dce))

    def test_dim_fil(self):
        ''' Accessor dim with different partitioning for FIL. '''
        # Adjacent, BATP upon OFMP.
        ps = PartitionScheme(order=[pe.INPP, pe.OUTP, pe.BATP, pe.OFMP],
                             pdims=[(2, 2), (5, 5), (3, 3), (7, 7)])
        self.assertTupleEqual(BufShrScheme(ps).dim(de.FIL), (15,) * 2)
        # Adjacent, OFMP upon BATP.
        ps = PartitionScheme(order=[pe.INPP, pe.OFMP, pe.BATP, pe.OUTP],
                             pdims=[(2, 2), (5, 5), (3, 3), (7, 7)])
        self.assertTupleEqual(BufShrScheme(ps).dim(de.FIL), (15,) * 2)

        # Not adjacent, BATP upon OFMP.
        ps = PartitionScheme(order=[pe.OUTP, pe.BATP, pe.INPP, pe.OFMP],
                             pdims=[(2, 2), (5, 5), (3, 3), (7, 7)])
        self.assertTupleEqual(BufShrScheme(ps).dim(de.FIL), (5,) * 2)
        # Not adjacent, OFMP upon BATP.
        ps = PartitionScheme(order=[pe.OFMP, pe.INPP, pe.BATP, pe.OUTP],
                             pdims=[(2, 2), (5, 5), (3, 3), (7, 7)])
        self.assertTupleEqual(BufShrScheme(ps).dim(de.FIL), (3,) * 2)

        # Only BATP.
        ps = PartitionScheme(order=[pe.OUTP, pe.BATP, pe.INPP, pe.OFMP],
                             pdims=[(2, 2), (1, 1), (3, 3), (7, 7)])
        self.assertTupleEqual(BufShrScheme(ps).dim(de.FIL), (3,) * 2)
        # Only OFMP.
        ps = PartitionScheme(order=[pe.OFMP, pe.INPP, pe.BATP, pe.OUTP],
                             pdims=[(2, 2), (5, 5), (1, 1), (7, 7)])
        self.assertTupleEqual(BufShrScheme(ps).dim(de.FIL), (5,) * 2)

    def test_dim_invalid_index(self):
        ''' Accessor dim invalid index. '''
        with self.assertRaises(IndexError):
            _ = self.bufshr1.dim(de.NUM)

    def test_size_invalid_index(self):
        ''' Get size invalid index. '''
        with self.assertRaises(IndexError):
            _ = self.bufshr1.size(de.NUM)

    def test_nbr_dists(self):
        ''' Accessor nbr_dists. '''
        self.assertTupleEqual(self.bufshr1.nbr_dists[de.FIL], (5, 2))
        self.assertTupleEqual(self.bufshr1.nbr_dists[de.IFM], (15, 2))
        self.assertTupleEqual(self.bufshr1.nbr_dists[de.OFM], (1, 1))

        self.assertTupleEqual(self.bufshr2.nbr_dists[de.FIL], (1, 1))
        self.assertTupleEqual(self.bufshr2.nbr_dists[de.IFM], (15, 15))
        self.assertTupleEqual(self.bufshr2.nbr_dists[de.OFM], (1, 1))

        self.assertTupleEqual(self.bufshr3.nbr_dists[de.FIL], (3, 5))
        self.assertTupleEqual(self.bufshr3.nbr_dists[de.IFM], (12, 10))
        self.assertTupleEqual(self.bufshr3.nbr_dists[de.OFM], (1, 1))

    def test_nhops_rotate_all(self):
        ''' Get nhops_rotate_all. '''
        # With `self.bufshr3` and FIL, the dimension is 4 by 2, with neighbor
        # distances 3 and 5.
        bufshr = self.bufshr3
        dce = de.FIL
        self.assertTupleEqual(bufshr.dim(dce), (4, 2))
        self.assertTupleEqual(bufshr.nbr_dists[dce], (3, 5))

        # Subgroup as 4 by 2. The whole circle is six hops of 3 and two hops of
        # 5, but only 7 of 8 steps.
        self.assertAlmostEqual(bufshr.nhops_rotate_all(dce, 8),
                               (3 * 6 + 5 * 2) * 7 / 8.)
        # Subgroup as 4 by 1. One node does three hops of 3, and other three
        # nodes do two hops of 3 and one hop of 9 (looping back).
        self.assertAlmostEqual(bufshr.nhops_rotate_all(dce, 4),
                               ((3 * 3) + (3 * 2 + 9) * 3) / 4. * 2)
        # Subgroup as 2 by 1. All nodes do one hop of 3.
        self.assertAlmostEqual(bufshr.nhops_rotate_all(dce, 2),
                               (3 + 3) / 2. * 4)
        # Subgroup as 1. No rotation.
        self.assertAlmostEqual(bufshr.nhops_rotate_all(dce, 1), 0)

        # Subgroup as 4 by 1. One node does two hops of 3 and two do one hop of
        # 3 and 6 each. The 3rd node also sends to the 4th one with two hops of
        # 3.
        self.assertAlmostEqual(bufshr.nhops_rotate_all(dce, 3),
                               ((3 * 2) + (3 + 6) * 2 + (3 * 2)) / 3. * 2)
        # Subgroup as 4 by 2. The 1st node does three hops of 3 and one hop of
        # 5. The 2nd, 3rd, and 4th nodes do two hops of 3, and one hop of 5,
        # and one looping back from the 5th node to the 1st node. The 5th node
        # does one looping back and three hops of 3. Finally, the 5th node also
        # sends to the 6th to 8th nodes.
        self.assertAlmostEqual(bufshr.nhops_rotate_all(dce, 5),
                               ((3 * 3 + 5) + (3 * 2 + 5 + (3 * 3 + 5)) * 3
                                + ((3 * 3 + 5) + 3 * 3) + 3 * 3 * 4) / 5.)
        # The others are similar.
        self.assertAlmostEqual(bufshr.nhops_rotate_all(dce, 6),
                               ((3 * 4 + 5) + (3 * 3 + 5 + (3 * 2 + 5)) * 4
                                + ((3 * 2 + 5) + 3 * 4) + 3 * 2 * 5) / 6.)
        self.assertAlmostEqual(bufshr.nhops_rotate_all(dce, 7),
                               ((3 * 5 + 5) + (3 * 4 + 5 + (3 * 1 + 5)) * 5
                                + ((3 * 1 + 5) + 3 * 5) + 3 * 1 * 6) / 7.)

    def test_nhops_rotate_all_invalid(self):
        ''' Get nhops_rotate_all with invalid args. '''
        with self.assertRaisesRegexp(ValueError, 'BufShrScheme: .*subgroup.*'):
            _ = self.bufshr3.nhops_rotate_all(
                de.FIL, self.bufshr3.size(de.FIL) + 1)

    def test_nhops_wide_fetch_once(self):
        ''' Get nhops_wide_fetch_once. '''
        # With `self.bufshr3` and FIL, the dimension is 4 by 2, with neighbor
        # distances 3 and 5.
        bufshr = self.bufshr3
        dce = de.FIL
        self.assertTupleEqual(bufshr.dim(dce), (4, 2))
        self.assertTupleEqual(bufshr.nbr_dists[dce], (3, 5))

        for subgrp_size in range(bufshr.size(dce)):
            self.assertAlmostEqual(
                bufshr.nhops_wide_fetch_once(dce, subgrp_size, 1), 0)

        # Three nodes fetch one hop of 3, and the last node fetches one hop of
        # 9 (looping back).
        self.assertAlmostEqual(bufshr.nhops_wide_fetch_once(dce, 4, 2),
                               (3 * 3 + 9) / 4. * 2)
        # Two nodes fetch one hop of 3, and the 3rd node fetches one hop of 6
        # (looping back). The last node fetches one hop of 3 from the 3rd.
        self.assertAlmostEqual(bufshr.nhops_wide_fetch_once(dce, 3, 2),
                               (3 * 2 + 6 + 3) / 3. * 2)
        # All nodes do one hop of 3.
        self.assertAlmostEqual(bufshr.nhops_wide_fetch_once(dce, 2, 2),
                               (3 + 3) / 2. * 4)

        for subgrp_size in range(2, bufshr.size(dce)):
            self.assertAlmostEqual(
                bufshr.nhops_wide_fetch_once(dce, subgrp_size, 1.5),
                bufshr.nhops_wide_fetch_once(dce, subgrp_size, 2) / 2.)

    def test_nhops_wide_fetch_once_inv(self):
        ''' Get nhops_wide_fetch_once with invalid args. '''
        with self.assertRaisesRegexp(ValueError, 'BufShrScheme: .*subgroup.*'):
            _ = self.bufshr3.nhops_wide_fetch_once(
                de.FIL, self.bufshr3.size(de.FIL) + 1, 2)

        with self.assertRaisesRegexp(ValueError, 'BufShrScheme: .*width.*'):
            _ = self.bufshr3.nhops_wide_fetch_once(
                de.FIL,
                self.bufshr3.size(de.FIL) / 2,
                self.bufshr3.size(de.FIL) / 2 + 1)

    def test_repr(self):
        ''' __repr__. '''
        self.assertIn(repr(self.ps1), repr(self.bufshr1))
        self.assertIn(repr(self.ps2), repr(self.bufshr2))
        self.assertIn(repr(self.ps3), repr(self.bufshr3))

