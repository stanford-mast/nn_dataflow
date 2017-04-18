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

'''
Utilities.
'''

import numpy as np

def idivc(valx, valy):
    '''
    Integer division and ceiling.

    Return the min integer that is no less than `valx / valy`.
    '''
    return (valx + valy - 1) // valy


def approx_dividable(total, num, overhead=0.2):
    ''' Whether it is reasonable to divide `total` into `num` parts.
    `overhead` is the allowed max padding overhead.  '''
    return idivc(total, num) * num < total * (1 + overhead)


def factorize(value, num, limits=None):
    '''
    Factorize given `value` into `num` numbers. Return as a copy of num-length
    np.array.

    Iterate over factor combinations of which the product is `value`.

    `limits` is a (num-1)-length tuple, specifying the upper limits for the
    first num-1 factors.
    '''
    if limits is None:
        limits = [float('inf')] * (num - 1)
    assert len(limits) >= num - 1
    limits = limits[:num-1] + [float('inf')]

    factors = np.ones(num, dtype=int)
    while True:
        # Calculate the last factor.
        factors[-1] = idivc(value, np.prod(factors[:-1]))
        if np.prod(factors) == value \
                and np.all(np.less(factors, limits)):
            yield tuple(np.copy(factors))

        # Update the first n - 1 factor combination, backwards.
        lvl = num - 1
        while lvl >= 0:
            factors[lvl] += 1
            if np.prod(factors[:lvl+1]) <= value:
                break
            else:
                factors[lvl] = 1
                lvl -= 1
        if lvl < 0:
            return


def closest_factor(value, factor):
    '''
    Return the maximum factor of `value` that is no larger than `factor`, and
    the minimum factor of `value` that is no less than `factor`, as a tuple.
    '''
    res = tuple()

    # Maximum no-larger factor.
    f = int(factor)
    while f > 1:
        if value % f == 0 and f <= factor:
            break
        f -= 1
    res += (max(1, f), )

    # Minimum no-smaller factor.
    f = int(factor)
    while f < value:
        if f != 0 and value % f == 0 and f >= factor:
            break
        f += 1
    res += (min(value, f), )

    return res


def assert_float_eq_int(vfloat, vint, message=''):
    '''
    Check the given float value is equal to the given int value. Print the
    optional message if not equal.
    '''
    if abs(vfloat - vint) > 1:
        raise AssertionError(message + ' {} != {}'.format(vfloat, vint))

