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

import unittest

from nn_dataflow.core import Option

class TestOption(unittest.TestCase):
    ''' Tests for Option. '''

    def test_valid_kwargs(self):
        ''' Valid keyword arguments. '''
        options = Option(sw_gbuf_bypass=(False, False, False),
                         sw_solve_loopblocking=False,
                         hw_access_forwarding=False,
                         hw_gbuf_sharing=False,
                         partition_hybrid=True,
                         partition_batch=False,
                         partition_ifmaps=False,
                         partition_interlayer=False,
                         opt_goal='ed',
                         ntops=10,
                         nprocesses=16,
                         verbose=False
                        )
        self.assertEqual(options.sw_gbuf_bypass, (False, False, False),
                         'sw_gbuf_bypass')
        self.assertEqual(options.sw_solve_loopblocking, False,
                         'sw_solve_loopblocking')
        self.assertEqual(options.hw_access_forwarding, False,
                         'hw_access_forwarding')
        self.assertEqual(options.hw_gbuf_sharing, False,
                         'hw_gbuf_sharing')
        self.assertEqual(options.partition_hybrid, True,
                         'partition_hybrid')
        self.assertEqual(options.partition_batch, False,
                         'partition_batch')
        self.assertEqual(options.partition_ifmaps, False,
                         'partition_ifmaps')
        self.assertEqual(options.partition_interlayer, False,
                         'partition_interlayer')
        self.assertEqual(options.opt_goal, 'ed', 'opt_goal')
        self.assertEqual(options.ntops, 10, 'ntops')
        self.assertEqual(options.nprocesses, 16, 'nprocesses')
        self.assertEqual(options.verbose, False, 'verbose')

    def test_valid_args(self):
        ''' Valid arguments. '''
        options = Option((False, True, False), True)
        self.assertEqual(options.sw_gbuf_bypass, (False, True, False),
                         'sw_gbuf_bypass')
        self.assertEqual(options.sw_solve_loopblocking, True,
                         'sw_solve_loopblocking')

    def test_default_args(self):
        ''' Default arguments. '''
        options = Option()
        self.assertTupleEqual(options.sw_gbuf_bypass, (False, False, False))
        self.assertEqual(options.sw_solve_loopblocking, False)
        self.assertEqual(options.partition_hybrid, False)
        self.assertEqual(options.partition_batch, False)
        self.assertEqual(options.partition_ifmaps, False)
        self.assertEqual(options.opt_goal, 'e')
        self.assertEqual(options.ntops, 1)
        self.assertEqual(options.nprocesses, 1)
        self.assertEqual(options.verbose, False)

    def test_invalid_args(self):
        ''' Invalid args. '''
        with self.assertRaisesRegex(TypeError, 'Option: .*at most.*100'):
            _ = Option(*[None] * 100)

    def test_invalid_kwargs(self):
        ''' Invalid kwargs. '''
        with self.assertRaisesRegex(TypeError, 'Option: .*bad.*'):
            _ = Option(bad='')

    def test_invalid_both_args_kwargs(self):
        ''' Invalid both args and kwargs are given. '''
        with self.assertRaisesRegex(TypeError, 'Option: .*sw_gbuf_bypass.*'):
            _ = Option((False,) * 3, sw_gbuf_bypass=(False,) * 3)

    def test_invalid_swgbyp_type(self):
        ''' Invalid sw_gbuf_bypass type. '''
        with self.assertRaisesRegex(TypeError, 'Option: .*sw_gbuf_bypass.*'):
            _ = Option(sw_gbuf_bypass=[False, False, False])

    def test_invalid_swgbyp_len(self):
        ''' Invalid sw_gbuf_bypass len. '''
        with self.assertRaisesRegex(ValueError, 'Option: .*sw_gbuf_bypass.*'):
            _ = Option(sw_gbuf_bypass=(False, False))

    def test_invalid_swsol_hwbufshr(self):
        ''' Invalid sw_solve_loopblocking and hw_gbuf_sharing comb. '''
        with self.assertRaisesRegex(ValueError,
                                    'Option: .*sw_solve_loopblocking.*'
                                    'hw_gbuf_sharing.*'):
            _ = Option(sw_solve_loopblocking=True, hw_gbuf_sharing=True)

    def test_invalid_hwaccfwd_hwbufshr(self):
        ''' Invalid hw_access_forwarding and hw_gbuf_sharing comb. '''
        with self.assertRaisesRegex(ValueError,
                                    'Option: .*hw_access_forwarding.*'
                                    'hw_gbuf_sharing.*'):
            _ = Option(hw_access_forwarding=True, hw_gbuf_sharing=True)

    def test_invalid_swsol_hwswb(self):
        ''' Invalid sw_solve_loopblocking and hw_gbuf_save_writeback comb. '''
        with self.assertRaisesRegex(ValueError,
                                    'Option: .*sw_solve_loopblocking.*'
                                    'hw_gbuf_save_writeback.*'):
            _ = Option(sw_solve_loopblocking=True, hw_gbuf_save_writeback=True)

    def test_invalid_part_hybrid_ifmaps(self):
        ''' Invalid partition_hybrid and partition_ifmaps comb. '''
        with self.assertRaisesRegex(ValueError,
                                    'Option: .*partition_ifmaps.*'
                                    'partition_hybrid.*'):
            _ = Option(partition_hybrid=False, partition_ifmaps=True)

    def test_invalid_time_ovhd(self):
        ''' Invalid layer_pipeline_time_ovhd. '''
        with self.assertRaisesRegex(KeyError,
                                    'Option: .*layer_pipeline_time_ovhd.*'):
            _ = Option(layer_pipeline_time_ovhd=None)

        with self.assertRaisesRegex(ValueError,
                                    'Option: .*layer_pipeline_time_ovhd.*'):
            _ = Option(layer_pipeline_time_ovhd=-1)

    def test_invalid_max_degree(self):
        ''' Invalid layer_pipeline_max_degree. '''
        with self.assertRaisesRegex(KeyError,
                                    'Option: .*layer_pipeline_max_degree.*'):
            _ = Option(layer_pipeline_max_degree=None)

        with self.assertRaisesRegex(ValueError,
                                    'Option: .*layer_pipeline_max_degree.*'):
            _ = Option(layer_pipeline_max_degree=-1)

    def test_invalid_opt_goal(self):
        ''' Invalid opt_goal. '''
        with self.assertRaisesRegex(ValueError, 'Option: .*opt_goal.*'):
            _ = Option(opt_goal='o')
        with self.assertRaisesRegex(ValueError, 'Option: .*opt_goal.*'):
            _ = Option(opt_goal='E')

    def test_option_list(self):
        ''' Accessor option_list. '''
        options = Option()
        self.assertCountEqual(options.option_list(), options._fields)

