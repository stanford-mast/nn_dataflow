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

import itertools
from collections import namedtuple

from .. import util
from .phy_dim2 import PhyDim2

NODE_REGION_LIST = ['dim',
                    'origin',
                    'dist',
                    'type',
                    'wtot',
                    'wbeg',
                   ]

class NodeRegion(namedtuple('NodeRegion', NODE_REGION_LIST)):
    '''
    A node region defined by the dimension and origin offset.

    The `type` attribute specifies the region type, which could be `PROC` for
    computation processing nodes or 'DRAM' for off-chip data storage nodes.

    The node region can be optionally folded along the w dimension in a zig-zag
    manner. The folding scheme is defined by (wtot, wbeg). `wtot` is always
    positive, representing the number of nodes between two turns (total width).
    `wbeg` is the number of nodes before reaching the first turning boundary,
    with its sign representing the direction. E.g.,

    ...
    ******************
              ********
              | wbeg |

    or

    ...
    ******************
    *********
    | -wbeg |

    With folded region, `origin` points to the first node.

    NOTE: we cannot overload __contains__ and __iter__ as a node container,
    because the base namedtuple already defines them.
    '''

    # Type enums.
    PROC = 0
    DRAM = 1
    NUM = 2

    def __new__(cls, *args, **kwargs):

        # Set default values.
        kwargs2 = kwargs.copy()
        if len(args) <= NODE_REGION_LIST.index('dist'):
            kwargs2.setdefault('dist', PhyDim2(1, 1))
        if len(args) <= NODE_REGION_LIST.index('wtot'):
            # Default to dim.w but we haven't checked dim yet. Replace later.
            kwargs2.setdefault('wtot', None)
        if len(args) <= NODE_REGION_LIST.index('wbeg'):
            # Default to wtot. Also replace later.
            kwargs2.setdefault('wbeg', None)

        ntp = super(NodeRegion, cls).__new__(cls, *args, **kwargs2)

        if not isinstance(ntp.dim, PhyDim2):
            raise TypeError('NodeRegion: dim must be a PhyDim2 object.')
        if not isinstance(ntp.origin, PhyDim2):
            raise TypeError('NodeRegion: origin must be a PhyDim2 object.')
        if not isinstance(ntp.dist, PhyDim2):
            raise TypeError('NodeRegion: dist must be a PhyDim2 object.')

        if ntp.type not in range(cls.NUM):
            raise ValueError('NodeRegion: type must be a valid type enum.')

        if ntp.wtot is None:
            ntp = ntp._replace(wtot=ntp.dim.w)
        if ntp.wbeg is None:
            ntp = ntp._replace(wbeg=ntp.wtot)

        if not isinstance(ntp.wtot, int):
            raise TypeError('NodeRegion: wtot must be an int.')
        if not isinstance(ntp.wbeg, int):
            raise TypeError('NodeRegion: wbeg must be an int.')

        if not (0 < abs(ntp.wbeg) <= ntp.wtot) and ntp.dim.size() > 0:
            raise ValueError('NodeRegion: |wbeg| must be in (0, wtot].')

        return ntp

    def contains_node(self, coordinate):
        ''' Whether the region contains the given absolute node coordinate. '''
        return coordinate in self.iter_node()

    def iter_node(self):
        ''' Iterate through all absolute node coordinates in the region. '''
        for rel_coord in itertools.product(*[range(d) for d in self.dim]):
            yield self.rel2abs(PhyDim2(*rel_coord))

    def rel2abs(self, rel_coordinate):
        ''' Convert relative node coordinate to absolute node coordinate. '''
        if not isinstance(rel_coordinate, PhyDim2):
            raise TypeError('NodeRegion: relative coordinate must be '
                            'a PhyDim2 object.')
        if not all(0 <= c < d for c, d in zip(rel_coordinate, self.dim)):
            raise ValueError('NodeRegion: relative coordinate {} is not in '
                             'node region {}.'.format(rel_coordinate, self))

        # Add starting offset to start from the boundary before the first node,
        # then modulo wtot to get the delta h and w to this boundary point.
        h, w = divmod(rel_coordinate.w + self.wtot - abs(self.wbeg), self.wtot)
        # Direction for w, changing every time when h increments.
        direction = (-1 if self.wbeg < 0 else 1) * (-1 if h % 2 else 1)
        # Make w relative to the left boundary.
        w = w if direction > 0 else self.wtot - 1 - w

        abs_coordinate = self.origin \
                + PhyDim2(h=h * self.dim.h + rel_coordinate.h,
                          w=w - (self.wtot - self.wbeg if self.wbeg > 0
                                 else -self.wbeg - 1)) \
                * self.dist

        return abs_coordinate

    def allocate(self, request_list):
        '''
        Allocate node subregions spatially within the node region according to
        the given `request_list` which is a list of numbers of nodes requested.

        Return a list of NodeRegion instances, whose origins are absolute
        offset (not relative to the origin of self). The allocation may fail if
        and only if the total number of nodes requested is larger than the
        number of nodes in the region, in which case an empty list is returned.

        The strategy is to allocate stripe-wise in a zig-zag order, allowing
        for folding in width. We first determine a stripe height as the
        greatest common divisor of the requested numbers of nodes. Then
        allocate each request as (stripe height, request size / stripe height)
        to fill in the stripe, and move to the next stripe after the current
        one is filled. If the width of a request is larger than the remaining
        width of the current stripe, we use up the remaining width, and fold
        the request width to the next stripe.
        '''

        if sum(request_list) > self.dim.size():
            return []

        hstrp = util.gcd(self.dim.h, *request_list)
        subregions = []

        wtot = self.dim.w
        ofs_h, ofs_w = 0, 0
        move_right = True

        for req in request_list:

            # Subregion.
            assert req % hstrp == 0
            width = req // hstrp

            subdim = PhyDim2(hstrp, width)
            if move_right:
                origin = PhyDim2(ofs_h, ofs_w)
                wbeg = min(wtot - ofs_w, width)
                assert wbeg > 0
            else:
                origin = PhyDim2(ofs_h, self.dim.w - ofs_w - 1)
                wbeg = -min(wtot - ofs_w, width)
                assert wbeg < 0

            subregions.append(NodeRegion(dim=subdim,
                                         origin=self.origin \
                                            + origin * self.dist,
                                         dist=self.dist,
                                         type=self.type,
                                         wtot=wtot,
                                         wbeg=wbeg))

            # Move the offset
            ofs_w += width
            while ofs_w >= self.dim.w:
                # Overflow, fold to the next stripe.
                ofs_w -= self.dim.w
                ofs_h += hstrp
                move_right = not move_right

        # Not moved outside the region.
        assert ofs_h + hstrp <= self.dim.h or ofs_w == 0

        return subregions

