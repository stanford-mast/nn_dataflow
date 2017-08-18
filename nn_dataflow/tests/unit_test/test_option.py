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

import unittest

from nn_dataflow.core import Option

class TestOption(unittest.TestCase):
    ''' Tests for Option. '''

    def test_valid_kwargs(self):
        ''' Valid keyword arguments. '''
        options = Option(sw_gbuf_bypass=(False, False, False),
                         sw_solve_loopblocking=False,
                         partition_hybrid=True,
                         partition_batch=False,
                         partition_ifmaps=False,
                         ntops=10,
                         nprocesses=16,
                         verbose=False
                        )
        self.assertEqual(options.sw_gbuf_bypass, (False, False, False),
                         'sw_gbuf_bypass')
        self.assertEqual(options.sw_solve_loopblocking, False,
                         'sw_solve_loopblocking')
        self.assertEqual(options.partition_hybrid, True,
                         'partition_hybrid')
        self.assertEqual(options.partition_batch, False,
                         'partition_batch')
        self.assertEqual(options.partition_ifmaps, False,
                         'partition_ifmaps')
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
        self.assertEqual(options.ntops, 1)
        self.assertEqual(options.nprocesses, 1)
        self.assertEqual(options.verbose, False)

    def test_invalid_args(self):
        ''' Invalid args. '''
        with self.assertRaisesRegexp(TypeError, 'Option: .*at most.*100'):
            _ = Option(*[None] * 100)

    def test_invalid_kwargs(self):
        ''' Invalid kwargs. '''
        with self.assertRaisesRegexp(TypeError, 'Option: .*bad.*'):
            _ = Option(bad='')

    def test_invalid_both_args_kwargs(self):
        ''' Invalid both args and kwargs are given. '''
        with self.assertRaisesRegexp(TypeError, 'Option: .*sw_gbuf_bypass.*'):
            _ = Option((False,) * 3, sw_gbuf_bypass=(False,) * 3)

    def test_invalid_swgbyp_type(self):
        ''' Invalid sw_gbuf_bypass type. '''
        with self.assertRaisesRegexp(TypeError, 'Option: .*sw_gbuf_bypass.*'):
            _ = Option(sw_gbuf_bypass=[False, False, False])

    def test_invalid_swgbyp_len(self):
        ''' Invalid sw_gbuf_bypass len. '''
        with self.assertRaisesRegexp(ValueError, 'Option: .*sw_gbuf_bypass.*'):
            _ = Option(sw_gbuf_bypass=(False, False))

    def test_invalid_part_hybrid_ifmaps(self):
        ''' Invalid partition_hybrid and partition_ifmaps comb. '''
        with self.assertRaisesRegexp(ValueError,
                                     'Option: .*partition_ifmaps.*'
                                     'partition_hybrid.*'):
            _ = Option(partition_hybrid=False, partition_ifmaps=True)

    def test_option_list(self):
        ''' Accessor option_list. '''
        options = Option()
        self.assertItemsEqual(options.option_list(), options._fields)

