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

from . import data_category_enum as de

OPTION_LIST = ['sw_gbuf_bypass',
               'sw_solve_loopblocking',
               'hw_access_forwarding',
               'hw_gbuf_sharing',
               'hw_gbuf_save_writeback',
               'partition_hybrid',
               'partition_batch',
               'partition_ifmaps',
               'partition_interlayer',
               'layer_pipeline_time_ovhd',
               'layer_pipeline_max_degree',
               'layer_pipeline_opt',
               'opt_goal',
               'ntops',
               'nprocesses',
               'verbose',
              ]

class Option(namedtuple('Option', OPTION_LIST)):
    '''
    Schedule options.
    '''

    def __new__(cls, *args, **kwargs):

        if len(args) > len(OPTION_LIST):
            raise TypeError('Option: can take at most {} arguments ({} given).'
                            .format(len(OPTION_LIST), len(args)))

        if not set(kwargs).issubset(OPTION_LIST):
            raise TypeError('Option: got an unexpected keyword argument {}.'
                            .format(next(k for k in kwargs
                                         if k not in OPTION_LIST)))

        # Combine args and kwargs.
        kwdict = kwargs.copy()
        for k, v in zip(OPTION_LIST, args):
            if k in kwdict:
                raise TypeError('Option: got multiple values for '
                                'keyword argument {}.'
                                .format(k))
            kwdict[k] = v

        kwdict.setdefault('sw_gbuf_bypass', (False,) * de.NUM)
        kwdict.setdefault('sw_solve_loopblocking', False)
        kwdict.setdefault('hw_access_forwarding', False)
        kwdict.setdefault('hw_gbuf_sharing', False)
        kwdict.setdefault('hw_gbuf_save_writeback', False)
        kwdict.setdefault('partition_hybrid', False)
        kwdict.setdefault('partition_batch', False)
        kwdict.setdefault('partition_ifmaps', False)
        kwdict.setdefault('partition_interlayer', False)
        kwdict.setdefault('layer_pipeline_time_ovhd', float('inf'))
        kwdict.setdefault('layer_pipeline_max_degree', float('inf'))
        kwdict.setdefault('layer_pipeline_opt', True)
        kwdict.setdefault('opt_goal', 'e')
        kwdict.setdefault('ntops', 1)
        kwdict.setdefault('nprocesses', 1)
        kwdict.setdefault('verbose', False)

        assert set(kwdict) == set(OPTION_LIST)

        ntp = super(Option, cls).__new__(cls, **kwdict)

        if not isinstance(ntp.sw_gbuf_bypass, tuple):
            raise TypeError('Option: sw_gbuf_bypass must be a tuple')
        if len(ntp.sw_gbuf_bypass) != de.NUM:
            raise ValueError('Option: sw_gbuf_bypass must have length {}'
                             .format(de.NUM))

        if ntp.sw_solve_loopblocking and ntp.hw_gbuf_sharing:
            raise ValueError('Option: sw_solve_loopblocking and '
                             'hw_gbuf_sharing cannot be simultaneously '
                             'enabled.')

        if ntp.hw_access_forwarding and ntp.hw_gbuf_sharing:
            raise ValueError('Option: hw_access_forwarding is implied by '
                             'hw_gbuf_sharing, thus cannot be both enabled.')

        if ntp.sw_solve_loopblocking and ntp.hw_gbuf_save_writeback:
            raise ValueError('Option: sw_solve_loopblocking and '
                             'hw_gbuf_save_writeback cannot be simultaneously '
                             'enabled.')

        if ntp.partition_ifmaps and not ntp.partition_hybrid:
            raise ValueError('Option: partition_ifmaps requires '
                             'partition_hybrid to be set.')

        if not isinstance(ntp.layer_pipeline_time_ovhd, (int, float)):
            raise KeyError('Option: layer_pipeline_time_ovhd must be a '
                           'number.')
        if ntp.layer_pipeline_time_ovhd < 0:
            raise ValueError('Option: layer_pipeline_time_ovhd must be '
                             'positive.')

        if not isinstance(ntp.layer_pipeline_max_degree, (int, float)):
            raise KeyError('Option: layer_pipeline_max_degree must be a '
                           'number.')
        if ntp.layer_pipeline_max_degree < 0:
            raise ValueError('Option: layer_pipeline_max_degree must be '
                             'positive.')

        if ntp.opt_goal not in ['e', 'd', 'ed']:
            raise ValueError('Option: opt_goal is invalid, must be one of '
                             '\'e\', \'d\', and \'ed\'.')

        return ntp

    @staticmethod
    def option_list():
        ''' List of options. '''
        return OPTION_LIST

