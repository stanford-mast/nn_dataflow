'''
Physical 2D dimension.
'''

from collections import namedtuple
from operator import add, sub, mul

class PhyDim2(namedtuple('PhyDim2', ['h', 'w'])):
    '''
    Denote a physical 2D dimension.
    '''

    def size(self):
        ''' Total size. '''
        return int(reduce(mul, self, 1))

    def __add__(self, other):
        ''' Return element-wise `self + other`. '''
        if not isinstance(other, PhyDim2):
            other = PhyDim2(other, other)
        return PhyDim2(map(add, self, other))

    def __sub__(self, other):
        ''' Return element-wise `self - other`. '''
        if not isinstance(other, PhyDim2):
            other = PhyDim2(other, other)
        return PhyDim2(map(sub, self, other))

    def __mul__(self, other):
        ''' Return element-wise `self * other`. '''
        if not isinstance(other, PhyDim2):
            other = PhyDim2(other, other)
        return PhyDim2(map(mul, self, other))

    __rmul__ = __mul__


