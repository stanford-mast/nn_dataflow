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

from nn_dataflow.core import DataCategoryEnum as de
from nn_dataflow.core import loop_blocking_solver
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow.core import Option

from . import TestLoopBlockingFixture

class TestLoopBlockingSolver(TestLoopBlockingFixture):
    ''' Tests for loop_blocking_solver module. '''

    def setUp(self):

        super(TestLoopBlockingSolver, self).setUp()

        # Bypass solver for each reside data category.
        self.optkeys_bypsol = ['BYPSOL_{}'.format(dce) for dce in range(de.NUM)]

        for reside_dce in range(de.NUM):
            opt_dict = self.options['BYPSOL']._asdict()
            byp = [True] * de.NUM
            byp[reside_dce] = False
            opt_dict['sw_gbuf_bypass'] = tuple(byp)

            self.options[self.optkeys_bypsol[reside_dce]] = Option(**opt_dict)

    def test_reside_sol(self):
        ''' Data reside solution. '''

        for reside_dce in range(de.NUM):

            optkey = self.optkeys_bypsol[reside_dce]

            for bl_ts, bl_ords \
                    in loop_blocking_solver.gen_loopblocking_gbuf_reside(
                            self.nld['BASE'], self.resource['BASE'],
                            self.options[optkey]):

                lbs = self._lbs(bl_ts, bl_ords, optkey=optkey)

                self.assertTrue(lbs.stored_in_gbuf[reside_dce])
                self.assertFalse(any(lbs.stored_in_gbuf[dce]
                                     for dce in range(de.NUM)
                                     if dce != reside_dce))

    def test_reside_sol_opt(self, rsrckey='BASE', wlkey='BASE'):
        ''' Data reside solution optimal. '''

        def _cost(lbs):
            access = lbs.get_access()
            return [int(sum(access[me.DRAM])), int(sum(access[me.GBUF]))]

        min_sch_dict = {}
        sol_sch_dict = {}

        # Among all schemes that bypass all non-reside data categories.
        for bl_ts, bl_ords in self._gen_loopblocking_all(wlkey=wlkey):

            lbs = self._lbs(bl_ts, bl_ords, wlkey=wlkey, rsrckey=rsrckey,
                            optkey='BYP')
            if not lbs.is_valid():
                continue

            all_reside_dce = [dce for dce in range(de.NUM)
                              if lbs.stored_in_gbuf[dce]]
            # Only look at the cases with one or none reside data category.
            if not all_reside_dce:
                min_sch = min_sch_dict.get(None, None)
                if not min_sch or _cost(lbs) < min_sch:
                    min_sch_dict[None] = _cost(lbs)
            elif len(all_reside_dce) == 1:
                dce, = all_reside_dce
                min_sch = min_sch_dict.get(dce, None)
                if not min_sch or _cost(lbs) < min_sch:
                    min_sch_dict[dce] = _cost(lbs)

        # Solve each reside data category.
        for reside_dce in range(de.NUM):

            optkey = self.optkeys_bypsol[reside_dce]

            for bl_ts, bl_ords \
                    in loop_blocking_solver.gen_loopblocking_gbuf_reside(
                            self.nld[wlkey], self.resource[rsrckey],
                            self.options[optkey]):

                lbs = self._lbs(bl_ts, bl_ords, wlkey=wlkey, rsrckey=rsrckey,
                                optkey='BYP')
                self.assertTrue(lbs.is_valid())
                self.assertFalse(any(lbs.stored_in_gbuf[dce]
                                     for dce in range(de.NUM)
                                     if dce != reside_dce))

                true_reside_dce = reside_dce \
                        if lbs.stored_in_gbuf[reside_dce] else None

                sol_sch = sol_sch_dict.get(true_reside_dce, None)
                if not sol_sch or _cost(lbs) < sol_sch:
                    sol_sch_dict[true_reside_dce] = _cost(lbs)

        self.assertTrue(sol_sch_dict.items() <= min_sch_dict.items(),
                        'test_reside_sol_opt: wlkey {} rsrckey {}: '
                        'solutions do not cover all optimal ones. '
                        'sol {} opt {}.'
                        .format(wlkey, rsrckey, sol_sch_dict, min_sch_dict))

        self.assertListEqual(
            min(sol_sch_dict.values()), min(min_sch_dict.values()),
            'test_reside_sol_opt: wlkey {} rsrckey {}: '
            'solutions do not cover the optimal one. sol {} opt {}.'
            .format(wlkey, rsrckey, sol_sch_dict, min_sch_dict))

    def test_reside_sol_opt_resource(self):
        ''' Data reside solution optimal with different resources. '''

        for rsrckey in ['LG', 'SM']:

            self.test_reside_sol_opt(rsrckey=rsrckey)

    def test_reside_sol_opt_pool(self):
        ''' Data reside solution optimal with PoolingLayer. '''

        with self.assertRaisesRegex(ValueError, 'loop_blocking_solver: .*'):
            self.test_reside_sol_opt(wlkey='POOL')

    def test_reside_sol_opt_zero(self):
        ''' Data reside solution optimal with zero size. '''

        for wlkey in ['ZERO_FIL', 'ZERO_IFM']:

            self.test_reside_sol_opt(wlkey=wlkey)

    def test_reside_sol_cnt(self):
        ''' Data reside solution count. '''

        all_set = set(loop_blocking_solver.gen_loopblocking_gbuf_reside(
            self.nld['BASE'], self.resource['BASE'], self.options['BYPSOL']))

        union_set = set()
        reside_set_list = []

        for reside_dce in range(de.NUM):

            optkey = self.optkeys_bypsol[reside_dce]

            s = set(loop_blocking_solver.gen_loopblocking_gbuf_reside(
                self.nld['BASE'], self.resource['BASE'], self.options[optkey]))

            reside_set_list.append(s)
            union_set |= s

        self.assertSetEqual(all_set, union_set)
        self.assertEqual(len(union_set), sum(len(s) for s in reside_set_list))

