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

from nn_dataflow import NestedLoopDesc
from nn_dataflow import DataCategoryEnum as de
from nn_dataflow import MemHierEnum as me

class TestNestedLoopDesc(unittest.TestCase):
    ''' Tests for NestedLoopDesc. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        nld = NestedLoopDesc(loopcnt_ifm=3,
                             loopcnt_ofm=8,
                             loopcnt_bat=4,
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             unit_ops=7,
                             unit_time=7
                            )
        self.assertEqual(nld.loopcnt_ifm, 3, 'loopcnt_ifm')
        self.assertEqual(nld.loopcnt_ofm, 8, 'loopcnt_ofm')
        self.assertEqual(nld.loopcnt_bat, 4, 'loopcnt_bat')
        self.assertEqual(nld.usize_gbuf, (20, 30, 9), 'usize_gbuf')
        self.assertEqual(nld.usize_regf, (3, 3, 1), 'usize_regf')
        self.assertEqual(nld.unit_access, ((19, 29, 9),
                                           (18, 28, 8),
                                           (35, 45, 15),
                                           (1, 1, 2)), 'unit_access')
        self.assertEqual(nld.unit_ops, 7, 'unit_ops')
        self.assertEqual(nld.unit_time, 7, 'unit_time')

    def test_invalid_usize_gbuf_type(self):
        ''' Invalid usize_gbuf type. '''
        with self.assertRaisesRegexp(TypeError,
                                     'NestedLoopDesc: .*usize_gbuf.*'):
            _ = NestedLoopDesc(loopcnt_ifm=3,
                               loopcnt_ofm=8,
                               loopcnt_bat=4,
                               usize_gbuf=[20, 30, 9],
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_usize_gbuf_len(self):
        ''' Invalid usize_gbuf len. '''
        with self.assertRaisesRegexp(ValueError,
                                     'NestedLoopDesc: .*usize_gbuf.*'):
            _ = NestedLoopDesc(loopcnt_ifm=3,
                               loopcnt_ofm=8,
                               loopcnt_bat=4,
                               usize_gbuf=(20, 30),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_usize_regf_type(self):
        ''' Invalid usize_regf type. '''
        with self.assertRaisesRegexp(TypeError,
                                     'NestedLoopDesc: .*usize_regf.*'):
            _ = NestedLoopDesc(loopcnt_ifm=3,
                               loopcnt_ofm=8,
                               loopcnt_bat=4,
                               usize_gbuf=(20, 30, 9),
                               usize_regf=[3, 3, 1],
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_usize_regf_len(self):
        ''' Invalid usize_regf len. '''
        with self.assertRaisesRegexp(ValueError,
                                     'NestedLoopDesc: .*usize_regf.*'):
            _ = NestedLoopDesc(loopcnt_ifm=3,
                               loopcnt_ofm=8,
                               loopcnt_bat=4,
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_unit_access_type(self):
        ''' Invalid unit_access type. '''
        with self.assertRaisesRegexp(TypeError,
                                     'NestedLoopDesc: .*unit_access.*'):
            _ = NestedLoopDesc(loopcnt_ifm=3,
                               loopcnt_ofm=8,
                               loopcnt_bat=4,
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=[(19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)],
                               unit_ops=7,
                               unit_time=7
                              )
        with self.assertRaisesRegexp(TypeError,
                                     'NestedLoopDesc: .*unit_access.*'):
            _ = NestedLoopDesc(loopcnt_ifm=3,
                               loopcnt_ofm=8,
                               loopcnt_bat=4,
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            [35, 45, 15],
                                            (1, 1, 2)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_unit_access_len(self):
        ''' Invalid unit_access len. '''
        with self.assertRaisesRegexp(ValueError,
                                     'NestedLoopDesc: .*unit_access.*'):
            _ = NestedLoopDesc(loopcnt_ifm=3,
                               loopcnt_ofm=8,
                               loopcnt_bat=4,
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               unit_ops=7,
                               unit_time=7
                              )
        with self.assertRaisesRegexp(ValueError,
                                     'NestedLoopDesc: .*unit_access.*'):
            _ = NestedLoopDesc(loopcnt_ifm=3,
                               loopcnt_ofm=8,
                               loopcnt_bat=4,
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_usize_gbuf_of(self):
        ''' Accessor usize_gbuf. '''
        nld = NestedLoopDesc(loopcnt_ifm=3,
                             loopcnt_ofm=8,
                             loopcnt_bat=4,
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             unit_ops=7,
                             unit_time=7
                            )
        self.assertEqual(nld.usize_gbuf_of(de.FIL), 20, 'usize_gbuf: FIL')
        self.assertEqual(nld.usize_gbuf_of(de.IFM), 30, 'usize_gbuf: IFM')
        self.assertEqual(nld.usize_gbuf_of(de.OFM), 9, 'usize_gbuf: OFM')

    def test_usize_regf_of(self):
        ''' Accessor usize_regf. '''
        nld = NestedLoopDesc(loopcnt_ifm=3,
                             loopcnt_ofm=8,
                             loopcnt_bat=4,
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             unit_ops=7,
                             unit_time=7
                            )
        self.assertEqual(nld.usize_regf_of(de.FIL), 3, 'usize_regf: FIL')
        self.assertEqual(nld.usize_regf_of(de.IFM), 3, 'usize_regf: IFM')
        self.assertEqual(nld.usize_regf_of(de.OFM), 1, 'usize_regf: OFM')

    def test_unit_access_at_of(self):
        ''' Accessor unit_access. '''
        nld = NestedLoopDesc(loopcnt_ifm=3,
                             loopcnt_ofm=8,
                             loopcnt_bat=4,
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             unit_ops=7,
                             unit_time=7
                            )
        self.assertEqual(nld.unit_access_at_of(me.DRAM), 19 + 29 + 9,
                         'unit_access: DRAM')
        self.assertEqual(nld.unit_access_at_of(me.ITCN), 35 + 45 + 15,
                         'unit_access: ITCN')
        self.assertEqual(nld.unit_access_at_of(me.GBUF, de.OFM), 8,
                         'unit_access: GBUF, OFM')
        self.assertEqual(nld.unit_access_at_of(me.REGF, de.FIL), 1,
                         'unit_access: REGF, FIL')

