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

from collections import namedtuple
import numbers

class IntRange(namedtuple('IntRange', ['beg', 'end'])):
    '''
    A range of integer numbers.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(IntRange, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.beg, numbers.Integral):
            raise TypeError('IntRange: begin value must be an integer.')
        if not isinstance(ntp.end, numbers.Integral):
            raise TypeError('IntRange: end value must be an integer.')
        if ntp.beg > ntp.end:
            raise ValueError('IntRange: begin value {} > end value {}?'
                             .format(ntp.beg, ntp.end))

        return ntp

    def size(self):
        '''
        Get the size of the range.
        '''
        return self.end - self.beg

    def empty(self):
        '''
        Whether the range is empty.
        '''
        return self.beg == self.end

    def range(self):
        '''
        Generator for the range.
        '''
        for v in range(self.beg, self.end):
            yield v

    def overlap(self, other):
        '''
        Get the overlapped IntRange of the two.
        '''
        if not isinstance(other, IntRange):
            raise TypeError('IntRange: an IntRange object is required.')
        try:
            return IntRange(max(self.beg, other.beg), min(self.end, other.end))
        except ValueError:
            # Non-overlapped.
            return IntRange(0, 0)

    def offset(self, val):
        '''
        Get a new IntRange by offseting `val`.
        '''
        return IntRange(self.beg + val, self.end + val)

