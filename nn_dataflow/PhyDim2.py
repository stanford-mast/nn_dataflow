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

from collections import namedtuple
from operator import add, sub, neg, mul

class PhyDim2(namedtuple('PhyDim2', ['h', 'w'])):
    '''
    Denote a physical 2D dimension.
    '''

    def size(self):
        ''' Total size. '''
        return int(reduce(mul, self, 1))

    def hop_dist(self, other):
        ''' Hop distance between twn coordinate. '''
        if not isinstance(other, PhyDim2):
            raise TypeError('PhyDim2: hop_dist only applies on two PhyDim2 '
                            'instances.')
        return abs(self.h - other.h) + abs(self.w - other.w)

    def __add__(self, other):
        ''' Return element-wise `self + other`. '''
        if not isinstance(other, PhyDim2):
            other = PhyDim2(other, other)
        return PhyDim2(*map(add, self, other))

    def __sub__(self, other):
        ''' Return element-wise `self - other`. '''
        if not isinstance(other, PhyDim2):
            other = PhyDim2(other, other)
        return PhyDim2(*map(sub, self, other))

    def __neg__(self):
        ''' Return element-wise negative. '''
        return PhyDim2(*map(neg, self))

    def __mul__(self, other):
        ''' Return element-wise `self * other`. '''
        if not isinstance(other, PhyDim2):
            other = PhyDim2(other, other)
        return PhyDim2(*map(mul, self, other))

    __rmul__ = __mul__

