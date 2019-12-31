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

import math
import unittest

from nn_dataflow import util

class TestUtilHashableDict(unittest.TestCase):
    ''' Tests for util.HashableDict. '''

    def test_fromdict(self):
        ''' fromdict. '''
        d = {'k': 1, 3: 'a'}

        hd1 = util.HashableDict.fromdict(d)
        self.assertSetEqual(set(d.items()), set(hd1.items()))

        hd2 = util.HashableDict.fromdict(d)
        self.assertNotEqual(id(hd1), id(hd2))
        self.assertEqual(hd1, hd2)
        self.assertEqual(hash(hd1), hash(hd2))

        hd3 = util.HashableDict.fromdict(
            d, keyfunc=str, valfunc=lambda x: frozenset([x]))
        self.assertNotEqual(hd1, hd3)

    def test_fromdict_error(self):
        ''' fromdict error. '''
        with self.assertRaisesRegex(TypeError, 'HashableDict: .*dict.*'):
            _ = util.HashableDict.fromdict([1, 2])

    def test_eq(self):
        ''' __eq__ and __ne__. '''
        hd = util.HashableDict([('k', 1), (3, 'a')])
        lst = ['k', 3]
        self.assertEqual(hd, hd.copy())
        self.assertNotEqual(hd, lst)

    def test_copy(self):
        ''' copy. '''
        hd = util.HashableDict([('k', 1), (3, 'a')])
        self.assertNotEqual(id(hd), id(hd.copy()))
        self.assertEqual(hd, hd.copy())
        self.assertEqual(hash(hd), hash(hd.copy()))

    def test_setitem_delitem(self):
        ''' __setitem__ and __delitem__. '''
        hd = util.HashableDict([('k', 1), (3, 'a')])

        self.assertIn('k', hd)
        self.assertEqual(hd[3], 'a')
        self.assertEqual(len(hd), 2)

        with self.assertRaises(KeyError):
            hd[2] = 'b'
        with self.assertRaises(KeyError):
            hd[3] = 'b'
        with self.assertRaises(KeyError):
            hd.update([(2, 'b')])
        with self.assertRaises(KeyError):
            hd.setdefault(2, [])

        with self.assertRaises(KeyError):
            del hd[3]
        with self.assertRaises(KeyError):
            hd.pop(3)
        with self.assertRaises(KeyError):
            hd.popitem()
        with self.assertRaises(KeyError):
            hd.clear()


class TestUtilIdivc(unittest.TestCase):
    ''' Tests for util.idivc. '''

    def test_int(self):
        ''' Int. '''
        self.assertEqual(util.idivc(8, 3), 3)
        self.assertEqual(util.idivc(8, 2), 4)
        self.assertEqual(util.idivc(8, 1), 8)

    def test_negative(self):
        ''' Negative. '''
        self.assertEqual(util.idivc(34, 4), 9, 'idivc: negative')
        self.assertEqual(util.idivc(-34, 4), -8, 'idivc: negative')
        self.assertEqual(util.idivc(34, -4), -8, 'idivc: negative')
        self.assertEqual(util.idivc(-34, -4), 9, 'idivc: negative')

    def test_zero(self):
        ''' Zero. '''
        self.assertEqual(util.idivc(0, 3), 0, 'idivc: zero')
        with self.assertRaises(ZeroDivisionError):
            _ = util.idivc(3, 0)

    def test_float(self):
        ''' Float. '''
        self.assertAlmostEqual(util.idivc(4.3, 3), 2)
        self.assertAlmostEqual(util.idivc(34.3, 3), 12)
        self.assertAlmostEqual(util.idivc(34, 3.), 12)

    def test_inf(self):
        ''' Inf. '''
        self.assertEqual(util.idivc(3, float('inf')), 0, 'idivc: inf')
        self.assertTrue(math.isnan(util.idivc(float('inf'), float('inf'))),
                        'idivc: inf')


class TestUtilProd(unittest.TestCase):
    ''' Tests for util.prod. '''

    def test_int(self):
        ''' Int. '''
        self.assertIsInstance(util.prod([3, 5, 7]), int)

        self.assertEqual(util.prod([3, 5, 7]), 105)
        self.assertEqual(util.prod([3, 5, -1]), -15)
        self.assertEqual(util.prod([3, -5, 7]), -105)
        self.assertEqual(util.prod([3, -5, 0]), 0)

        self.assertEqual(util.prod((3, 5, 7)), 105)
        self.assertEqual(util.prod(set([3, 5, 7])), 105)
        self.assertEqual(util.prod({3: 'a', 5: 'b', 7: 'c'}), 105)

    def test_float(self):
        ''' Float. '''
        self.assertAlmostEqual(util.prod([1.1, 2, 3]), 6.6)
        self.assertAlmostEqual(util.prod([1.1, 2, -3.]), -6.6)

    def test_empty(self):
        ''' Empty. '''
        self.assertEqual(util.prod([]), 1)
        self.assertEqual(util.prod(tuple()), 1)
        self.assertEqual(util.prod(set()), 1)


class TestUtilApproxDividable(unittest.TestCase):
    ''' Tests for util.approx_dividable. '''

    def test_int(self):
        ''' Int. '''
        self.assertTrue(util.approx_dividable(24, 2,
                                              rel_overhead=0, abs_overhead=0))
        self.assertTrue(util.approx_dividable(24, 3,
                                              rel_overhead=0, abs_overhead=0))
        self.assertTrue(util.approx_dividable(24, 4,
                                              rel_overhead=0, abs_overhead=0))

        self.assertTrue(util.approx_dividable(11, 2))
        self.assertFalse(util.approx_dividable(8, 5))
        self.assertTrue(util.approx_dividable(19, 5))

        self.assertTrue(util.approx_dividable(7, 2,
                                              rel_overhead=0.2,
                                              abs_overhead=0))
        self.assertTrue(util.approx_dividable(7, 2,
                                              rel_overhead=0,
                                              abs_overhead=1))
        self.assertTrue(util.approx_dividable(19, 7,
                                              rel_overhead=0.2,
                                              abs_overhead=0))
        self.assertTrue(util.approx_dividable(19, 7,
                                              rel_overhead=0,
                                              abs_overhead=2))
        self.assertFalse(util.approx_dividable(22, 7,
                                               rel_overhead=0.2,
                                               abs_overhead=0))
        self.assertFalse(util.approx_dividable(23, 7,
                                               rel_overhead=0,
                                               abs_overhead=1))

        ovhd = (21 - 19) / max(21., 19.)
        self.assertFalse(util.approx_dividable(19, 7,
                                               rel_overhead=ovhd - 0.01,
                                               abs_overhead=0))
        self.assertTrue(util.approx_dividable(19, 7,
                                              rel_overhead=ovhd + 0.01,
                                              abs_overhead=0))

    def test_float(self):
        ''' Float. '''
        self.assertTrue(util.approx_dividable(18.4, 3))
        self.assertTrue(util.approx_dividable(21.4, 3))


class TestUtilFactorize(unittest.TestCase):
    ''' Tests for util.factorize. '''

    def test_prod(self):
        ''' Check prod. '''
        for fs in util.factorize(24, 3):
            self.assertEqual(util.prod(fs), 24)

        for fs in util.factorize(1024, 3):
            self.assertEqual(util.prod(fs), 1024)

    def test_limits(self):
        ''' Check limits. '''
        for fs in util.factorize(1024, 3, limits=(10, 20)):
            self.assertLessEqual(fs[0], 10)
            self.assertLessEqual(fs[1], 20)
            self.assertEqual(util.prod(fs), 1024)

    def test_len(self):
        ''' Length. '''
        # Use 4 prime factors, 2, 3, 5, 7.
        val = 2 * 3 * 5 * 7
        self.assertEqual(len(list(util.factorize(val, 2))), 2 ** 4)
        self.assertEqual(len(list(util.factorize(val, 3))), 3 ** 4)

        for val in [24, 1024, (2 ** 4) * (3 ** 5) * (5 ** 2)]:
            fs = list(util.factorize(val, 2))
            self.assertEqual(len(fs), len(set(fs)))

    def test_factors(self):
        ''' Factors. '''
        factors2 = set()
        for fs in util.factorize(24, 2):
            factors2.update(fs)
        self.assertSetEqual(factors2, set([1, 2, 3, 4, 6, 8, 12, 24]))

        factors3 = set()
        for fs in util.factorize(24, 3):
            factors3.update(fs)
        self.assertSetEqual(factors2, factors3)

    def test_perm(self):
        ''' Permutations. '''
        fs_ord = set()
        fs_unord = set()
        for fs in util.factorize(512, 3):
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
    ''' Tests for util.closest_factor. '''

    def test_int(self):
        ''' Int. '''
        self.assertTupleEqual(util.closest_factor(24, 5), (4, 6))
        self.assertTupleEqual(util.closest_factor(24, 10), (8, 12))

        self.assertTupleEqual(util.closest_factor(25, 3), (1, 5))
        self.assertTupleEqual(util.closest_factor(25, 20), (5, 25))

    def test_exact(self):
        ''' Exact factor. '''
        self.assertTupleEqual(util.closest_factor(24, 6), (6, 6))
        self.assertTupleEqual(util.closest_factor(24, 2), (2, 2))
        self.assertTupleEqual(util.closest_factor(3, 1), (1, 1))

    def test_value_float(self):
        ''' Value is float. '''
        with self.assertRaisesRegex(TypeError, '.*integers.*'):
            _ = util.closest_factor(24.3, 5)
        with self.assertRaisesRegex(TypeError, '.*integers.*'):
            _ = util.closest_factor(24., 10)

    def test_factor_float(self):
        ''' Factor is float. '''
        self.assertTupleEqual(util.closest_factor(24, 5.3), (4, 6))
        self.assertTupleEqual(util.closest_factor(24, 10.2), (8, 12))

    def test_zero(self):
        ''' Zero. '''
        self.assertTupleEqual(util.closest_factor(0, 3), (3,))
        self.assertTupleEqual(util.closest_factor(24, 0), (1,))

    def test_negative(self):
        ''' Negative. '''
        with self.assertRaisesRegex(ValueError, '.*negative.*'):
            _ = util.closest_factor(24, -5)
        with self.assertRaisesRegex(ValueError, '.*negative.*'):
            _ = util.closest_factor(-24, -5)
        with self.assertRaisesRegex(ValueError, '.*negative.*'):
            _ = util.closest_factor(-24, 5)

    def test_missing(self):
        ''' Missing one or both. '''
        fs = util.closest_factor(4, 5)
        self.assertTupleEqual(fs, (4,))

        fs = util.closest_factor(4, 0.2)
        self.assertTupleEqual(fs, (1,))

    def test_random(self):
        ''' Random test. '''
        for val in range(1, 11):
            for f in range(1, 11):
                fs = util.closest_factor(val, f)
                string = 'closest_factor: {} {} {}'.format(val, f, fs)

                if len(fs) == 2:
                    self.assertEqual(val % fs[0], 0, string)
                    self.assertGreaterEqual(f, fs[0], string)
                    self.assertEqual(val % fs[1], 0, string)
                    self.assertLessEqual(f, fs[1], string)
                elif len(fs) == 1:
                    self.assertEqual(val % fs[0], 0, string)


class TestUtilGetIthRange(unittest.TestCase):
    ''' Tests for util.get_ith_range. '''

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
                beg, end = util.get_ith_range(rng, idx, num)
                self.assertEqual(beg, last_end)
                last_end = end
            self.assertEqual(last_end, rng[1])

    def test_equal_size(self):
        ''' Equal size. '''
        for rng, num in self.test_list:

            min_size = float('inf')
            max_size = -float('inf')
            for idx in range(num):
                beg, end = util.get_ith_range(rng, idx, num)
                min_size = min(min_size, end - beg)
                max_size = max(max_size, end - beg)
            self.assertLessEqual(max_size - min_size, 1)


class TestUtilGCD(unittest.TestCase):
    ''' Tests for util.gcd. '''

    def test_int(self):
        ''' Integers. '''
        self.assertEqual(util.gcd(3, 4), 1)
        self.assertEqual(util.gcd(8, 4), 4)
        self.assertEqual(util.gcd(3, 9), 3)
        self.assertEqual(util.gcd(15, 12), 3)
        self.assertEqual(util.gcd(300, 410), 10)

    def test_multi(self):
        ''' Multiple values. '''
        self.assertEqual(util.gcd(4, 8, 10), 2)
        self.assertEqual(util.gcd(*range(6, 21, 3)), 3)

    def test_single(self):
        ''' Single value. '''
        for v in range(1, 10):
            self.assertEqual(util.gcd(v), v)

    def test_no_arg(self):
        ''' No argument. '''
        with self.assertRaises(ValueError):
            _ = util.gcd()

    def test_float(self):
        ''' Float. '''
        with self.assertRaisesRegex(TypeError, '.*integers.*'):
            _ = util.gcd(1., 2)

        with self.assertRaisesRegex(TypeError, '.*integers.*'):
            _ = util.gcd(1, 2.2)

        with self.assertRaisesRegex(TypeError, '.*integers.*'):
            _ = util.gcd(1, 2, 3, 4.2)

    def test_non_positive(self):
        ''' Non-positive values. '''
        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.gcd(-1, 2)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.gcd(1, -2)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.gcd(3, 6, 9, 12, -21)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.gcd(3, 0)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.gcd(0, 3)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.gcd(0, 5, 10, 15, 20)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.gcd(5, 10, 0, 15, 20)


class TestUtilLCM(unittest.TestCase):
    ''' Tests for util.lcm. '''

    def test_int(self):
        ''' Integers. '''
        self.assertEqual(util.lcm(3, 4), 12)
        self.assertEqual(util.lcm(8, 4), 8)
        self.assertEqual(util.lcm(3, 9), 9)
        self.assertEqual(util.lcm(15, 12), 60)
        self.assertEqual(util.lcm(300, 410), 12300)

    def test_multi(self):
        ''' Multiple values. '''
        self.assertEqual(util.lcm(4, 8, 10), 40)
        self.assertEqual(util.lcm(*range(6, 21, 3)), 180)

    def test_single(self):
        ''' Single value. '''
        for v in range(1, 10):
            self.assertEqual(util.lcm(v), v)

    def test_no_arg(self):
        ''' No argument. '''
        with self.assertRaises(ValueError):
            _ = util.lcm()

    def test_float(self):
        ''' Float. '''
        with self.assertRaisesRegex(TypeError, '.*integers.*'):
            _ = util.lcm(1., 2)

        with self.assertRaisesRegex(TypeError, '.*integers.*'):
            _ = util.lcm(1, 2.2)

        with self.assertRaisesRegex(TypeError, '.*integers.*'):
            _ = util.lcm(1, 2, 3, 4.2)

    def test_non_positive(self):
        ''' Non-positive values. '''
        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.lcm(-1, 2)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.lcm(1, -2)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.lcm(3, 6, 9, 12, -21)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.lcm(3, 0)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.lcm(0, 3)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.lcm(0, 5, 10, 15, 20)

        with self.assertRaisesRegex(ValueError, '.*positive.*'):
            _ = util.lcm(5, 10, 0, 15, 20)


class TestUtilIsclose(unittest.TestCase):
    ''' Tests for util.isclose. '''

    def test_default_tol(self):
        ''' Default tolerants. '''
        self.assertTrue(util.isclose(14, 14))
        self.assertTrue(util.isclose(-19, -19))

        self.assertFalse(util.isclose(14, -14))
        self.assertFalse(util.isclose(2, 3))
        self.assertFalse(util.isclose(2, 2.01))

    def test_rel_tol(self):
        ''' Relative tolerant. '''
        self.assertTrue(util.isclose(14., 14.001, rel_tol=1e-3))
        self.assertTrue(util.isclose(0.001, 0.001001, rel_tol=1e-3))

        self.assertFalse(util.isclose(-14., 14.001, rel_tol=1e-3))
        self.assertFalse(util.isclose(0.001, 0.0011, rel_tol=1e-3))

    def test_abs_tol(self):
        ''' Absolute tolerant. '''
        self.assertTrue(util.isclose(14., 16, abs_tol=3))
        self.assertTrue(util.isclose(14., 14.001, abs_tol=2e-3))
        self.assertTrue(util.isclose(0.001, 0.001001, abs_tol=2e-6))
        self.assertTrue(util.isclose(0.001, 0.0011, abs_tol=2e-4))

        self.assertFalse(util.isclose(-14., 14.001, abs_tol=1))

    def test_both_tol(self):
        ''' Both tolerant. '''
        self.assertTrue(util.isclose(14., 14.001, rel_tol=1e-3, abs_tol=2e-6))
        self.assertTrue(util.isclose(14., 14.001, rel_tol=1e-6, abs_tol=2e-3))
        self.assertTrue(util.isclose(14., 14.001, rel_tol=1e-3, abs_tol=2e-3))
        self.assertFalse(util.isclose(14., 14.001, rel_tol=1e-6, abs_tol=2e-6))


class TestUtilAssertFloatEqInt(unittest.TestCase):
    ''' Tests for util.assert_float_eq_int. '''

    def test_success(self):
        ''' Success. '''
        # pylint: disable=no-self-use
        util.assert_float_eq_int(12., 12)
        util.assert_float_eq_int(12.3, 12)
        util.assert_float_eq_int(12.99, 12)
        util.assert_float_eq_int(11.01, 12)
        util.assert_float_eq_int(-11.8, -12)
        util.assert_float_eq_int(.01, 0)
        util.assert_float_eq_int(-.01, 0)

    def test_fail(self):
        ''' Fail. '''
        with self.assertRaisesRegex(AssertionError, '.*12.*'):
            util.assert_float_eq_int(13.01, 12)
        with self.assertRaisesRegex(AssertionError, '.*12.*'):
            util.assert_float_eq_int(10.99, 12)
        with self.assertRaisesRegex(AssertionError, '.*12.*'):
            util.assert_float_eq_int(12., -12)

