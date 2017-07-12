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

from operator import mul

'''
Utilities.
'''

class ContentHashClass(object):
    '''
    Class using the content instead of the object ID for hash.

    Such class instance can be used as key in dictionary.
    '''
    # pylint: disable=too-few-public-methods

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


def idivc(valx, valy):
    '''
    Integer division and ceiling.

    Return the min integer that is no less than `valx / valy`.
    '''
    return (valx + valy - 1) // valy


def prod(lst):
    ''' Get the product of a list. '''
    return reduce(mul, lst, 1)


def approx_dividable(total, num, overhead=0.2):
    ''' Whether it is reasonable to divide `total` into `num` parts.
    `overhead` is the allowed max padding overhead.  '''
    return idivc(total, num) * num <= total * (1 + overhead)


def factorize(value, num, limits=None):
    '''
    Factorize given `value` into `num` numbers. Return a tuple of length
    `num`.

    Iterate over factor combinations of which the product is `value`.

    `limits` is a (num-1)-length tuple, specifying the upper limits for the
    first num-1 factors.
    '''
    if limits is None:
        limits = [float('inf')] * (num - 1)
    assert len(limits) >= num - 1
    limits = list(limits[:num-1]) + [float('inf')]

    factors = [1] * num
    while True:
        # Calculate the last factor.
        factors[-1] = idivc(value, prod(factors[:-1]))
        if prod(factors) == value \
                and all(f <= l for f, l in zip(factors, limits)):
            yield tuple(factors)

        # Update the first n - 1 factor combination, backwards.
        lvl = num - 1
        while lvl >= 0:
            factors[lvl] += 1
            if prod(factors[:lvl+1]) <= value:
                break
            else:
                factors[lvl] = 1
                lvl -= 1
        if lvl < 0:
            return


def closest_factor(value, factor):
    '''
    Return the maximum factor of `value` that is no larger than `factor` (if
    any), and the minimum factor of `value` that is no less than `factor` (if
    any), as a tuple.
    '''
    if not isinstance(value, int):
        raise TypeError('value must be integers.')

    if value < 0 or factor < 0:
        raise ValueError('arguments must not be negative.')

    res = tuple()

    # Maximum no-larger factor.
    if factor >= 1:
        f = int(factor) + 1
        while f > factor:
            f -= 1
        while True:
            if f != 0 and value % f == 0:
                break
            f -= 1
        assert f <= factor and value % f == 0
        res += (f,)

    # Minimum no-smaller factor.
    if factor <= abs(value):
        f = int(factor) - 1
        while f < factor:
            f += 1
        while True:
            if f != 0 and value % f == 0:
                break
            f += 1
        assert f >= factor and value % f == 0
        res += (f,)

    return res


def get_ith_range(rng, idx, num):
    '''
    Divide the full range `rng` into `num` parts, and get the `idx`-th range.
    '''
    length = rng[1] - rng[0]
    beg = rng[0] + idx * length / num
    end = rng[0] + (idx + 1) * length / num
    assert end <= rng[1]
    return beg, end


def isclose(vala, valb, rel_tol=1e-9, abs_tol=0.0):
    '''
    Whether two values are close to each other.

    Identical to math.isclose() in Python 3.5.
    '''
    return abs(vala - valb) <= max(rel_tol * max(abs(vala), abs(valb)), abs_tol)


def assert_float_eq_int(vfloat, vint, message=''):
    '''
    Check the given float value is equal to the given int value. Print the
    optional message if not equal.
    '''
    if abs(vfloat - vint) > 1:
        raise AssertionError(message + ' {} != {}'.format(vfloat, vint))

