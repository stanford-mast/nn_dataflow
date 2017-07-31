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

import math
import unittest

from nn_dataflow import Util

class TestUtilIdivc(unittest.TestCase):
    ''' Tests for Util.idivc. '''

    def test_int(self):
        ''' Int. '''
        self.assertEqual(Util.idivc(8, 3), 3)
        self.assertEqual(Util.idivc(8, 2), 4)
        self.assertEqual(Util.idivc(8, 1), 8)

    def test_negative(self):
        ''' Negative. '''
        self.assertEqual(Util.idivc(34, 4), 9, 'idivc: negative')
        self.assertEqual(Util.idivc(-34, 4), -8, 'idivc: negative')
        self.assertEqual(Util.idivc(34, -4), -8, 'idivc: negative')
        self.assertEqual(Util.idivc(-34, -4), 9, 'idivc: negative')

    def test_zero(self):
        ''' Zero. '''
        self.assertEqual(Util.idivc(0, 3), 0, 'idivc: zero')
        with self.assertRaises(ZeroDivisionError):
            _ = Util.idivc(3, 0)

    def test_float(self):
        ''' Float. '''
        self.assertAlmostEqual(Util.idivc(4.3, 3), 2)
        self.assertAlmostEqual(Util.idivc(34.3, 3), 12)
        self.assertAlmostEqual(Util.idivc(34, 3.), 12)


class TestUtilProd(unittest.TestCase):
    ''' Tests for Util.prod. '''

    def test_int(self):
        ''' Int. '''
        self.assertIsInstance(Util.prod([3, 5, 7]), int)

        self.assertEqual(Util.prod([3, 5, 7]), 105)
        self.assertEqual(Util.prod([3, 5, -1]), -15)
        self.assertEqual(Util.prod([3, -5, 7]), -105)
        self.assertEqual(Util.prod([3, -5, 0]), 0)

        self.assertEqual(Util.prod((3, 5, 7)), 105)
        self.assertEqual(Util.prod(set([3, 5, 7])), 105)
        self.assertEqual(Util.prod({3: 'a', 5: 'b', 7: 'c'}), 105)

    def test_float(self):
        ''' Float. '''
        self.assertAlmostEqual(Util.prod([1.1, 2, 3]), 6.6)
        self.assertAlmostEqual(Util.prod([1.1, 2, -3.]), -6.6)

    def test_empty(self):
        ''' Empty. '''
        self.assertEqual(Util.prod([]), 1)
        self.assertEqual(Util.prod(tuple()), 1)
        self.assertEqual(Util.prod(set()), 1)


class TestUtilApproxDividable(unittest.TestCase):
    ''' Tests for Util.approx_dividable. '''

    def test_int(self):
        ''' Int. '''
        self.assertTrue(Util.approx_dividable(24, 2, overhead=0))
        self.assertTrue(Util.approx_dividable(24, 3, overhead=0))
        self.assertTrue(Util.approx_dividable(24, 4, overhead=0))

        self.assertTrue(Util.approx_dividable(7, 2))
        self.assertTrue(Util.approx_dividable(19, 7))
        self.assertFalse(Util.approx_dividable(22, 7))

        ovhd = Util.idivc(19, 7) * 7 / 19. - 1
        self.assertFalse(Util.approx_dividable(19, 7, overhead=ovhd - 0.01))
        self.assertTrue(Util.approx_dividable(19, 7, overhead=ovhd + 0.01))

    def test_float(self):
        ''' Float. '''
        self.assertTrue(Util.approx_dividable(18.4, 7))
        self.assertTrue(Util.approx_dividable(21.4, 7))


class TestUtilFactorize(unittest.TestCase):
    ''' Tests for Util.factorize. '''

    def test_prod(self):
        ''' Check prod. '''
        for fs in Util.factorize(24, 3):
            self.assertEqual(Util.prod(fs), 24)

        for fs in Util.factorize(1024, 3):
            self.assertEqual(Util.prod(fs), 1024)

    def test_limits(self):
        ''' Check limits. '''
        for fs in Util.factorize(1024, 3, limits=(10, 20)):
            self.assertLessEqual(fs[0], 10)
            self.assertLessEqual(fs[1], 20)
            self.assertEqual(Util.prod(fs), 1024)

    def test_len(self):
        ''' Length. '''
        # Use 4 prime factors, 2, 3, 5, 7.
        val = 2 * 3 * 5 * 7
        self.assertEqual(len(list(Util.factorize(val, 2))), 2 ** 4)
        self.assertEqual(len(list(Util.factorize(val, 3))), 3 ** 4)

        for val in [24, 1024, (2 ** 4) * (3 ** 5) * (5 ** 2)]:
            fs = list(Util.factorize(val, 2))
            self.assertEqual(len(fs), len(set(fs)))

    def test_factors(self):
        ''' Factors. '''
        factors2 = set()
        for fs in Util.factorize(24, 2):
            factors2.update(fs)
        self.assertSetEqual(factors2, set([1, 2, 3, 4, 6, 8, 12, 24]))

        factors3 = set()
        for fs in Util.factorize(24, 3):
            factors3.update(fs)
        self.assertSetEqual(factors2, factors3)

    def test_perm(self):
        ''' Permutations. '''
        fs_ord = set()
        fs_unord = set()
        for fs in Util.factorize(512, 3):
            fs_ord.add(fs)
            fs_unord.add(frozenset(fs))

        cnt = 0
        for fs in fs_unord:
            if len(fs) == 3:
                # Permutations.
                cnt += math.factorial(3)
            elif len(fs) == 2:
                # Permutations of a, a, b.
                cnt += 3
            else:
                # Pattern a, a, a.
                cnt += 1
        self.assertEqual(len(fs_ord), cnt)


class TestUtilClosestFactor(unittest.TestCase):
    ''' Tests for Util.closest_factor. '''

    def test_int(self):
        ''' Int. '''
        self.assertTupleEqual(Util.closest_factor(24, 5), (4, 6))
        self.assertTupleEqual(Util.closest_factor(24, 10), (8, 12))

        self.assertTupleEqual(Util.closest_factor(25, 3), (1, 5))
        self.assertTupleEqual(Util.closest_factor(25, 20), (5, 25))

    def test_exact(self):
        ''' Exact factor. '''
        self.assertTupleEqual(Util.closest_factor(24, 6), (6, 6))
        self.assertTupleEqual(Util.closest_factor(24, 2), (2, 2))
        self.assertTupleEqual(Util.closest_factor(3, 1), (1, 1))

    def test_value_float(self):
        ''' Value is float. '''
        with self.assertRaisesRegexp(TypeError, '.*integers.*'):
            _ = Util.closest_factor(24.3, 5)
        with self.assertRaisesRegexp(TypeError, '.*integers.*'):
            _ = Util.closest_factor(24., 10)

    def test_factor_float(self):
        ''' Factor is float. '''
        self.assertTupleEqual(Util.closest_factor(24, 5.3), (4, 6))
        self.assertTupleEqual(Util.closest_factor(24, 10.2), (8, 12))

    def test_zero(self):
        ''' Zero. '''
        self.assertTupleEqual(Util.closest_factor(0, 3), (3,))
        self.assertTupleEqual(Util.closest_factor(24, 0), (1,))

    def test_negative(self):
        ''' Negative. '''
        with self.assertRaisesRegexp(ValueError, '.*negative.*'):
            _ = Util.closest_factor(24, -5)
        with self.assertRaisesRegexp(ValueError, '.*negative.*'):
            _ = Util.closest_factor(-24, -5)
        with self.assertRaisesRegexp(ValueError, '.*negative.*'):
            _ = Util.closest_factor(-24, 5)

    def test_missing(self):
        ''' Missing one or both. '''
        fs = Util.closest_factor(4, 5)
        self.assertTupleEqual(fs, (4,))

        fs = Util.closest_factor(4, 0.2)
        self.assertTupleEqual(fs, (1,))

    def test_random(self):
        ''' Random test. '''
        for val in range(1, 11):
            for f in range(1, 11):
                fs = Util.closest_factor(val, f)
                string = 'closest_factor: {} {} {}'.format(val, f, fs)

                if len(fs) == 2:
                    self.assertEqual(val % fs[0], 0, string)
                    self.assertGreaterEqual(f, fs[0], string)
                    self.assertEqual(val % fs[1], 0, string)
                    self.assertLessEqual(f, fs[1], string)
                elif len(fs) == 1:
                    self.assertEqual(val % fs[0], 0, string)


class TestUtilGetIthRange(unittest.TestCase):
    ''' Tests for Util.get_ith_range. '''

    def setUp(self):
        self.test_list = [((0, 16), 4),
                          ((0, 44), 5),
                          ((5, 39), 7),
                          ((10, 41), 8),
                          ((10, 43), 8),
                         ]

    def test_coverage(self):
        ''' Coverage. '''
        for rng, num in self.test_list:

            last_end = rng[0]
            for idx in range(num):
                beg, end = Util.get_ith_range(rng, idx, num)
                self.assertEqual(beg, last_end)
                last_end = end
            self.assertEqual(last_end, rng[1])

    def test_equal_size(self):
        ''' Equal size. '''
        for rng, num in self.test_list:

            min_size = float('inf')
            max_size = -float('inf')
            for idx in range(num):
                beg, end = Util.get_ith_range(rng, idx, num)
                min_size = min(min_size, end - beg)
                max_size = max(max_size, end - beg)
            self.assertLessEqual(max_size - min_size, 1)


class TestUtilGCD(unittest.TestCase):
    ''' Tests for Util.gcd. '''

    def test_int(self):
        ''' Integers. '''
        self.assertEqual(Util.gcd(3, 4), 1)
        self.assertEqual(Util.gcd(8, 4), 4)
        self.assertEqual(Util.gcd(3, 9), 3)
        self.assertEqual(Util.gcd(15, 12), 3)
        self.assertEqual(Util.gcd(300, 410), 10)

    def test_multi(self):
        ''' Multiple values. '''
        self.assertEqual(Util.gcd(4, 8, 10), 2)
        self.assertEqual(Util.gcd(*range(6, 21, 3)), 3)

    def test_single(self):
        ''' Single value. '''
        for v in range(1, 10):
            self.assertEqual(Util.gcd(v), v)

    def test_no_arg(self):
        ''' No argument. '''
        with self.assertRaises(ValueError):
            _ = Util.gcd()

    def test_float(self):
        ''' Float. '''
        with self.assertRaisesRegexp(TypeError, '.*integers.*'):
            _ = Util.gcd(1., 2)

        with self.assertRaisesRegexp(TypeError, '.*integers.*'):
            _ = Util.gcd(1, 2.2)

        with self.assertRaisesRegexp(TypeError, '.*integers.*'):
            _ = Util.gcd(1, 2, 3, 4.2)

    def test_non_positive(self):
        ''' Non-positive values. '''
        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.gcd(-1, 2)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.gcd(1, -2)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.gcd(3, 6, 9, 12, -21)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.gcd(3, 0)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.gcd(0, 3)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.gcd(0, 5, 10, 15, 20)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.gcd(5, 10, 0, 15, 20)


class TestUtilLCM(unittest.TestCase):
    ''' Tests for Util.lcm. '''

    def test_int(self):
        ''' Integers. '''
        self.assertEqual(Util.lcm(3, 4), 12)
        self.assertEqual(Util.lcm(8, 4), 8)
        self.assertEqual(Util.lcm(3, 9), 9)
        self.assertEqual(Util.lcm(15, 12), 60)
        self.assertEqual(Util.lcm(300, 410), 12300)

    def test_multi(self):
        ''' Multiple values. '''
        self.assertEqual(Util.lcm(4, 8, 10), 40)
        self.assertEqual(Util.lcm(*range(6, 21, 3)), 180)

    def test_single(self):
        ''' Single value. '''
        for v in range(1, 10):
            self.assertEqual(Util.lcm(v), v)

    def test_no_arg(self):
        ''' No argument. '''
        with self.assertRaises(ValueError):
            _ = Util.lcm()

    def test_float(self):
        ''' Float. '''
        with self.assertRaisesRegexp(TypeError, '.*integers.*'):
            _ = Util.lcm(1., 2)

        with self.assertRaisesRegexp(TypeError, '.*integers.*'):
            _ = Util.lcm(1, 2.2)

        with self.assertRaisesRegexp(TypeError, '.*integers.*'):
            _ = Util.lcm(1, 2, 3, 4.2)

    def test_non_positive(self):
        ''' Non-positive values. '''
        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.lcm(-1, 2)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.lcm(1, -2)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.lcm(3, 6, 9, 12, -21)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.lcm(3, 0)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.lcm(0, 3)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.lcm(0, 5, 10, 15, 20)

        with self.assertRaisesRegexp(ValueError, '.*positive.*'):
            _ = Util.lcm(5, 10, 0, 15, 20)


class TestUtilIsclose(unittest.TestCase):
    ''' Tests for Util.isclose. '''

    def test_default_tol(self):
        ''' Default tolerants. '''
        self.assertTrue(Util.isclose(14, 14))
        self.assertTrue(Util.isclose(-19, -19))

        self.assertFalse(Util.isclose(14, -14))
        self.assertFalse(Util.isclose(2, 3))
        self.assertFalse(Util.isclose(2, 2.01))

    def test_rel_tol(self):
        ''' Relative tolerant. '''
        self.assertTrue(Util.isclose(14., 14.001, rel_tol=1e-3))
        self.assertTrue(Util.isclose(0.001, 0.001001, rel_tol=1e-3))

        self.assertFalse(Util.isclose(-14., 14.001, rel_tol=1e-3))
        self.assertFalse(Util.isclose(0.001, 0.0011, rel_tol=1e-3))

    def test_abs_tol(self):
        ''' Absolute tolerant. '''
        self.assertTrue(Util.isclose(14., 16, abs_tol=3))
        self.assertTrue(Util.isclose(14., 14.001, abs_tol=2e-3))
        self.assertTrue(Util.isclose(0.001, 0.001001, abs_tol=2e-6))
        self.assertTrue(Util.isclose(0.001, 0.0011, abs_tol=2e-4))

        self.assertFalse(Util.isclose(-14., 14.001, abs_tol=1))

    def test_both_tol(self):
        ''' Both tolerant. '''
        self.assertTrue(Util.isclose(14., 14.001, rel_tol=1e-3, abs_tol=2e-6))
        self.assertTrue(Util.isclose(14., 14.001, rel_tol=1e-6, abs_tol=2e-3))
        self.assertTrue(Util.isclose(14., 14.001, rel_tol=1e-3, abs_tol=2e-3))
        self.assertFalse(Util.isclose(14., 14.001, rel_tol=1e-6, abs_tol=2e-6))


class TestUtilAssertFloatEqInt(unittest.TestCase):
    ''' Tests for Util.assert_float_eq_int. '''

    def test_success(self):
        ''' Success. '''
        # pylint: disable=no-self-use
        Util.assert_float_eq_int(12., 12)
        Util.assert_float_eq_int(12.3, 12)
        Util.assert_float_eq_int(12.99, 12)
        Util.assert_float_eq_int(11.01, 12)
        Util.assert_float_eq_int(-11.8, -12)
        Util.assert_float_eq_int(.01, 0)
        Util.assert_float_eq_int(-.01, 0)

    def test_fail(self):
        ''' Fail. '''
        with self.assertRaisesRegexp(AssertionError, '.*12.*'):
            Util.assert_float_eq_int(13.01, 12)
        with self.assertRaisesRegexp(AssertionError, '.*12.*'):
            Util.assert_float_eq_int(10.99, 12)
        with self.assertRaisesRegexp(AssertionError, '.*12.*'):
            Util.assert_float_eq_int(12., -12)

