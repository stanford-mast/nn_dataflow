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

from . import DataCategoryEnum as de

OPTION_LIST = ['sw_gbuf_bypass',
               'sw_solve_loopblocking',
               'partition_hybrid',
               'partition_batch',
               'partition_ifmaps',
               'ntops',
               'nprocesses',
               'verbose',
              ]

class Option(namedtuple('Option', OPTION_LIST)):
    '''
    Schedule options.
    '''

    def __new__(cls, *args, **kwargs):
        ntp = super(Option, cls).__new__(cls, *args, **kwargs)

        if not isinstance(ntp.sw_gbuf_bypass, tuple):
            raise TypeError('Option: sw_gbuf_bypass must be a tuple')
        if len(ntp.sw_gbuf_bypass) != de.NUM:
            raise ValueError('Option: sw_gbuf_bypass must have length {}'
                             .format(de.NUM))

        if ntp.partition_ifmaps and not ntp.partition_hybrid:
            raise ValueError('Option: partition_ifmaps requires '
                             'partition_hybrid to be set.')

        return ntp

    @staticmethod
    def option_list():
        ''' List of options. '''
        return OPTION_LIST

