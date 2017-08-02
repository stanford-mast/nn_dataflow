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

from . import LoopEnum as le

class SchedulingConstraint(namedtuple('SchedulingConstraint',
                                      ['top_bl_t',
                                       'top_bl_lpe',
                                       'fmap_tpart',
                                      ])):
    '''
    Layer scheduling constraint.

    NOTE: the top blocking factors must be exactly equal, except for the BAT
    loop, which needs to be a multiple of the constrained factor because the
    mapping strategy may increase the effective batch size.
    '''

    def __new__(cls, top_bl_t=None, top_bl_lpe=None, fmap_tpart=1):
        ntp = super(SchedulingConstraint, cls).__new__(
            cls, top_bl_t=top_bl_t, top_bl_lpe=top_bl_lpe,
            fmap_tpart=fmap_tpart)

        if ntp.top_bl_t is not None:
            if not isinstance(ntp.top_bl_t, tuple):
                raise TypeError('SchedulingConstraint: top_bl_t must be None '
                                'or a tuple.')
            if len(ntp.top_bl_t) != le.NUM:
                raise ValueError('SchedulingConstraint: top_bl_t must have '
                                 'length {}.'.format(le.NUM))

        if ntp.top_bl_lpe is not None:
            if ntp.top_bl_lpe not in range(le.NUM):
                raise ValueError('SchedulingConstraint: top_bl_lpe must be '
                                 'None or a LoopEnum.')

        if not isinstance(ntp.fmap_tpart, int):
            raise TypeError('SchedulingConstraint: fmap_tpart must be an '
                            'integer.')

        return ntp

    def is_valid_top_bl(self, top_bl_t, top_bl_ord):
        '''
        The given `top_bl_t` and `top_bl_lpe` are valid with the constraint.
        '''
        # Check top_bl_t.
        if self.top_bl_t is not None:
            if any(self.top_bl_t[lpe] is not None
                   and not (self.top_bl_t[lpe] == top_bl_t[lpe] if lpe != le.BAT
                            else top_bl_t[lpe] % self.top_bl_t[lpe] == 0)
                   for lpe in range(le.NUM)):
                return False

        # Check top_bl_ord.
        if self.top_bl_lpe is not None and any(t > 1 for t in top_bl_t):
            top_bl_lpe = max(range(le.NUM), key=(lambda lpe: top_bl_ord[lpe]
                                                 if top_bl_t[lpe] > 1 else -1))
            if top_bl_lpe != self.top_bl_lpe:
                return False

        return True

