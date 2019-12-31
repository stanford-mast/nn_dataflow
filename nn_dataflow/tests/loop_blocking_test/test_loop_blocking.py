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

from nn_dataflow.core import loop_blocking
from nn_dataflow.core import DataCategoryEnum as de

from . import TestLoopBlockingFixture

class TestLoopBlocking(TestLoopBlockingFixture):
    ''' Tests for loop_blocking module. '''

    def test_skip_not_reg(self):
        ''' skip non-regularized. '''

        for sch in self._gen_loopblocking_all():

            skip = loop_blocking.skip_conv(*sch)
            reg_sch = self._regularized_scheme(*sch)

            if not skip:
                self.assertEqual(reg_sch, sch,
                                 'test_skip_not_reg: non-skipped {} should be '
                                 'regularized to {}'
                                 .format(sch, reg_sch))
                continue

            lbs = self._lbs(*sch, rsrckey='LG')
            reg_lbs = self._lbs(*reg_sch, rsrckey='LG')

            self.assertFalse(loop_blocking.skip_conv(*reg_sch),
                             'test_skip_not_reg: regularized {} is skipped.'
                             .format(reg_sch))
            self.assertAlmostEqual(lbs.get_access_cost(self.cost),
                                   reg_lbs.get_access_cost(self.cost),
                                   msg=('test_skip_not_reg: cost mismatch. '
                                        'orig {}, reg {}.'
                                        .format(sch, reg_sch)))
            self.assertListEqual(lbs.get_access(), reg_lbs.get_access(),
                                 msg=('test_skip_not_reg: access mismatch. '
                                      'orig {}, reg {}.'
                                      .format(sch, reg_sch)))
            size = self._get_lbs_size(lbs)
            reg_size = self._get_lbs_size(reg_lbs)
            self.assertTrue(all(all(ss1 >= ss2 for ss1, ss2 in zip(s1, s2))
                                for s1, s2 in zip(size, reg_size)),
                            'test_skip_not_reg: reg size is larger than eqv.\n'
                            'org {} has size {}\nreg {} has size {}'
                            .format(sch, size, reg_sch, reg_size))

    def test_skip_ratio(self):
        ''' skip ratio. '''

        cnts = [0, 0]

        for bl_ts, bl_ords in self._gen_loopblocking_all():

            skip = loop_blocking.skip_conv(bl_ts, bl_ords)
            cnts[skip] += 1

        skip_ratio = 1. * cnts[True] / sum(cnts)
        self.assertGreater(skip_ratio, 0.95,
                           'test_skip_ratio: skip ratio {} too low.'
                           .format(skip_ratio))

    def test_gen_loopblocking_all(self):
        ''' gen_loopblocking cover all. '''

        exp_cnt = 0
        for bl_ts, bl_ords in self._gen_loopblocking_all():
            exp_cnt += 1 if not loop_blocking.skip_conv(bl_ts, bl_ords) else 0

        cnt = 0
        for _ in self._gen_loopblocking(rsrckey='LG'):
            cnt += 1

        self.assertEqual(cnt, exp_cnt)

    def test_gen_loopblocking_mp(self):
        ''' gen_loopblocking multiprocessing. '''

        cnt1 = 0
        for _ in self._gen_loopblocking(rsrckey='LG'):
            cnt1 += 1

        cnt8 = 0
        for _ in self._gen_loopblocking(rsrckey='LG', optkey='MP'):
            cnt8 += 1

        self.assertEqual(cnt1, cnt8)

    def test_gen_loopblocking_no_eqv(self):
        ''' gen_loopblocking no equivalent. '''

        acc_dict = {}

        for lbs in self._gen_loopblocking(rsrckey='LG', skip_invalid=True):

            # Make the keys hashable (list -> tuple).
            size = tuple(tuple(ss for ss in s) for s in self._get_lbs_size(lbs))
            access = tuple(tuple(int(aa) for aa in a) for a in lbs.access)
            keys = (size, access)

            self.assertNotIn(keys, acc_dict,
                             'test_gen_loopblocking_no_eqv: found equivalents. '
                             'keys: access {} size {}'
                             .format(access, size))
            acc_dict[keys] = lbs

    def test_gen_loopblocking_ntops(self):
        ''' gen_loopblocking ntops. '''

        tops = list(self._gen_loopblocking(rsrckey='LG', optkey='NTOPS'))

        cost_prev = -float('inf')

        for lbs in self._gen_loopblocking(rsrckey='LG', skip_invalid=True):

            cost_curr = lbs.get_access_cost(self.cost)
            self.assertLessEqual(cost_prev, cost_curr)
            cost_prev = cost_curr

            if tops:
                top_lbs = tops.pop(0)
                self.assertAlmostEqual(cost_curr,
                                       top_lbs.get_access_cost(self.cost))

    def test_gen_loopblocking_byp_sol(self):
        ''' gen_loopblocking using bypass solvers. '''

        cnt = 0

        for lbs in self._gen_loopblocking(optkey='BYPSOL'):

            self.assertTrue(lbs.is_valid())

            cnt += 1

        self.assertLessEqual(cnt, 8)

    def test_gen_loopblocking_cstr(self):
        ''' gen_loopblocking with constraint. '''

        for lbs in self._gen_loopblocking(rsrckey='LG', cstr=self.cstr):

            self.assertTrue(self.cstr.is_valid_top_bl(lbs.bl_ts[0],
                                                      lbs.bl_ords[0]))

    def test_gen_loopblocking_cstr_sol(self):
        ''' gen_loopblocking using bypass solvers with constraint. '''

        cnt1 = len(list(self._gen_loopblocking(optkey='BYPSOL')))

        lbs_list = list(self._gen_loopblocking(optkey='BYPSOL', cstr=self.cstr))
        self.assertTrue(all(
            self.cstr.is_valid_top_bl(lbs.bl_ts[0], lbs.bl_ords[0])
            for lbs in lbs_list))
        cnt2 = len(lbs_list)

        self.assertLessEqual(cnt2, cnt1)

    def _gen_loopblocking(self, wlkey='BASE', rsrckey='BASE',
                          optkey='BASE', cstr=None, skip_invalid=False):
        ''' gen_loopblocking trampoline. '''
        if cstr is None:
            cstr = self.none_cstr
        for lbs in loop_blocking.gen_loopblocking(
                self.nld[wlkey], self.resource[rsrckey], self.part, cstr,
                self.cost, self.options[optkey]):
            if not skip_invalid or lbs.is_valid():
                yield lbs

    @staticmethod
    def _get_lbs_size(lbs):
        ''' Get the size info. '''
        assert lbs.is_valid()
        return [[lbs.data_size(bl, dce) for dce in range(de.NUM)]
                for bl in range(lbs.BL.NUM)]

