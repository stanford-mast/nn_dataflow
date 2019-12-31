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

from nn_dataflow.core import NestedLoopDesc
from nn_dataflow.core import DataDimLoops
from nn_dataflow.core import DataCategoryEnum as de
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow.core import LoopEnum as le

class TestNestedLoopDesc(unittest.TestCase):
    ''' Tests for NestedLoopDesc. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        nld = NestedLoopDesc(loopcnt=(3, 8, 4),
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             data_loops=(DataDimLoops(le.IFM, le.OFM),
                                         DataDimLoops(le.IFM, le.BAT),
                                         DataDimLoops(le.OFM, le.BAT)),
                             unit_ops=7,
                             unit_time=7
                            )
        self.assertEqual(nld.loopcnt, (3, 8, 4), 'loopcnt')
        self.assertEqual(nld.usize_gbuf, (20, 30, 9), 'usize_gbuf')
        self.assertEqual(nld.usize_regf, (3, 3, 1), 'usize_regf')
        self.assertEqual(nld.unit_access, ((19, 29, 9),
                                           (18, 28, 8),
                                           (35, 45, 15),
                                           (1, 1, 2)), 'unit_access')
        self.assertEqual(nld.data_loops[de.FIL], DataDimLoops(le.IFM, le.OFM),
                         'data_loops: FIL')
        self.assertEqual(nld.data_loops[de.IFM], DataDimLoops(le.IFM, le.BAT),
                         'data_loops: IFM')
        self.assertEqual(nld.data_loops[de.OFM], DataDimLoops(le.OFM, le.BAT),
                         'data_loops: OFM')
        self.assertEqual(nld.unit_ops, 7, 'unit_ops')
        self.assertEqual(nld.unit_time, 7, 'unit_time')

    def test_invalid_loopcnt_type(self):
        ''' Invalid loopcnt type. '''
        with self.assertRaisesRegex(TypeError,
                                    'NestedLoopDesc: .*loopcnt.*'):
            _ = NestedLoopDesc(loopcnt=[3, 8, 4],
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_loopcnt_len(self):
        ''' Invalid loopcnt len. '''
        with self.assertRaisesRegex(ValueError,
                                    'NestedLoopDesc: .*loopcnt.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8),
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_usize_gbuf_type(self):
        ''' Invalid usize_gbuf type. '''
        with self.assertRaisesRegex(TypeError,
                                    'NestedLoopDesc: .*usize_gbuf.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=[20, 30, 9],
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_usize_gbuf_len(self):
        ''' Invalid usize_gbuf len. '''
        with self.assertRaisesRegex(ValueError,
                                    'NestedLoopDesc: .*usize_gbuf.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=(20, 30),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_usize_regf_type(self):
        ''' Invalid usize_regf type. '''
        with self.assertRaisesRegex(TypeError,
                                    'NestedLoopDesc: .*usize_regf.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=(20, 30, 9),
                               usize_regf=[3, 3, 1],
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_usize_regf_len(self):
        ''' Invalid usize_regf len. '''
        with self.assertRaisesRegex(ValueError,
                                    'NestedLoopDesc: .*usize_regf.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_unit_access_type(self):
        ''' Invalid unit_access type. '''
        with self.assertRaisesRegex(TypeError,
                                    'NestedLoopDesc: .*unit_access.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=[(19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)],
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )
        with self.assertRaisesRegex(TypeError,
                                    'NestedLoopDesc: .*unit_access.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            [35, 45, 15],
                                            (1, 1, 2)),
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_unit_access_len(self):
        ''' Invalid unit_access len. '''
        with self.assertRaisesRegex(ValueError,
                                    'NestedLoopDesc: .*unit_access.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )
        with self.assertRaisesRegex(ValueError,
                                    'NestedLoopDesc: .*unit_access.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_usize_gbuf_of(self):
        ''' Accessor usize_gbuf. '''
        nld = NestedLoopDesc(loopcnt=(3, 8, 4),
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             data_loops=(DataDimLoops(le.IFM, le.OFM),
                                         DataDimLoops(le.IFM, le.BAT),
                                         DataDimLoops(le.OFM, le.BAT)),
                             unit_ops=7,
                             unit_time=7
                            )
        self.assertEqual(nld.usize_gbuf_of(de.FIL), 20, 'usize_gbuf: FIL')
        self.assertEqual(nld.usize_gbuf_of(de.IFM), 30, 'usize_gbuf: IFM')
        self.assertEqual(nld.usize_gbuf_of(de.OFM), 9, 'usize_gbuf: OFM')

    def test_usize_regf_of(self):
        ''' Accessor usize_regf. '''
        nld = NestedLoopDesc(loopcnt=(3, 8, 4),
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             data_loops=(DataDimLoops(le.IFM, le.OFM),
                                         DataDimLoops(le.IFM, le.BAT),
                                         DataDimLoops(le.OFM, le.BAT)),
                             unit_ops=7,
                             unit_time=7
                            )
        self.assertEqual(nld.usize_regf_of(de.FIL), 3, 'usize_regf: FIL')
        self.assertEqual(nld.usize_regf_of(de.IFM), 3, 'usize_regf: IFM')
        self.assertEqual(nld.usize_regf_of(de.OFM), 1, 'usize_regf: OFM')

    def test_unit_access_at_of(self):
        ''' Accessor unit_access. '''
        nld = NestedLoopDesc(loopcnt=(3, 8, 4),
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             data_loops=(DataDimLoops(le.IFM, le.OFM),
                                         DataDimLoops(le.IFM, le.BAT),
                                         DataDimLoops(le.OFM, le.BAT)),
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

    def test_invalid_data_loops_type(self):
        ''' Invalid data_loops type. '''
        with self.assertRaisesRegex(TypeError,
                                    'NestedLoopDesc: .*data_loops.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=[DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT),
                                           DataDimLoops(le.OFM, le.BAT)],
                               unit_ops=7,
                               unit_time=7
                              )
        with self.assertRaisesRegex(TypeError,
                                    'NestedLoopDesc: .*data_loops.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=((le.IFM, le.OFM),
                                           (le.IFM, le.BAT),
                                           (le.OFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_invalid_data_loops_len(self):
        ''' Invalid data_loops len. '''
        with self.assertRaisesRegex(ValueError,
                                    'NestedLoopDesc: .*data_loops.*'):
            _ = NestedLoopDesc(loopcnt=(3, 8, 4),
                               usize_gbuf=(20, 30, 9),
                               usize_regf=(3, 3, 1),
                               unit_access=((19, 29, 9),
                                            (18, 28, 8),
                                            (35, 45, 15),
                                            (1, 1, 2)),
                               data_loops=(DataDimLoops(le.IFM, le.OFM),
                                           DataDimLoops(le.IFM, le.BAT)),
                               unit_ops=7,
                               unit_time=7
                              )

    def test_total_ops(self):
        ''' Get total_ops. '''
        nld = NestedLoopDesc(loopcnt=(3, 8, 4),
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             data_loops=(DataDimLoops(le.IFM, le.OFM),
                                         DataDimLoops(le.IFM, le.BAT),
                                         DataDimLoops(le.OFM, le.BAT)),
                             unit_ops=7,
                             unit_time=7
                            )
        self.assertEqual(nld.total_ops(), 7 * 3 * 8 * 4)

    def test_total_access_of_at(self):
        ''' Get total_access_of_at. '''
        nld = NestedLoopDesc(loopcnt=(3, 8, 4),
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             data_loops=(DataDimLoops(le.IFM, le.OFM),
                                         DataDimLoops(le.IFM, le.BAT),
                                         DataDimLoops(le.OFM, le.BAT)),
                             unit_ops=7,
                             unit_time=7
                            )

        self.assertEqual(nld.total_access_at_of(me.DRAM, de.FIL), 19 * 3 * 8)
        self.assertEqual(nld.total_access_at_of(me.DRAM, de.IFM), 29 * 3 * 4)
        self.assertEqual(nld.total_access_at_of(me.DRAM, de.OFM), 9 * 8 * 4)

        self.assertEqual(nld.total_access_at_of(me.GBUF, de.FIL), 18 * 3 * 8)
        self.assertEqual(nld.total_access_at_of(me.GBUF, de.IFM), 28 * 3 * 4)
        self.assertEqual(nld.total_access_at_of(me.GBUF, de.OFM), 8 * 8 * 4)

        self.assertEqual(nld.total_access_at_of(me.ITCN, de.FIL), 35 * 3 * 8)
        self.assertEqual(nld.total_access_at_of(me.ITCN, de.IFM), 45 * 3 * 4)
        self.assertEqual(nld.total_access_at_of(me.ITCN, de.OFM), 15 * 8 * 4)

        self.assertEqual(nld.total_access_at_of(me.REGF, de.FIL), 1 * 3 * 8)
        self.assertEqual(nld.total_access_at_of(me.REGF, de.IFM), 1 * 3 * 4)
        self.assertEqual(nld.total_access_at_of(me.REGF, de.OFM), 2 * 8 * 4)

    def test_total_access_of_at_sum(self):
        ''' Get total_access_of_at sum. '''
        nld = NestedLoopDesc(loopcnt=(3, 8, 4),
                             usize_gbuf=(20, 30, 9),
                             usize_regf=(3, 3, 1),
                             unit_access=((19, 29, 9),
                                          (18, 28, 8),
                                          (35, 45, 15),
                                          (1, 1, 2)),
                             data_loops=(DataDimLoops(le.IFM, le.OFM),
                                         DataDimLoops(le.IFM, le.BAT),
                                         DataDimLoops(le.OFM, le.BAT)),
                             unit_ops=7,
                             unit_time=7
                            )

        self.assertEqual(nld.total_access_at_of(me.DRAM),
                         19 * 3 * 8 + 29 * 3 * 4 + 9 * 8 * 4)

        self.assertEqual(nld.total_access_at_of(me.GBUF),
                         18 * 3 * 8 + 28 * 3 * 4 + 8 * 8 * 4)

        self.assertEqual(nld.total_access_at_of(me.ITCN),
                         35 * 3 * 8 + 45 * 3 * 4 + 15 * 8 * 4)

        self.assertEqual(nld.total_access_at_of(me.REGF),
                         1 * 3 * 8 + 1 * 3 * 4 + 2 * 8 * 4)

