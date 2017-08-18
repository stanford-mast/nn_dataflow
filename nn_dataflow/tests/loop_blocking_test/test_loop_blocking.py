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

from nn_dataflow.core import loop_blocking

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
            self.assertAlmostEqual(lbs.get_cost(self.cost),
                                   reg_lbs.get_cost(self.cost),
                                   msg=('test_skip_not_reg: cost mismatch. '
                                        'orig {}, reg {}.'
                                        .format(sch, reg_sch)))
            self.assertListEqual(lbs.get_access(), reg_lbs.get_access(),
                                 msg=('test_skip_not_reg: access mismatch. '
                                      'orig {}, reg {}.'
                                      .format(sch, reg_sch)))
            size = lbs.get_scheme_dict(self.cost)['size']
            reg_size = reg_lbs.get_scheme_dict(self.cost)['size']
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
        for _ in loop_blocking.gen_loopblocking(
                self.nld['BASE'], self.resource['LG'], self.cost, 1,
                self.options['BASE']):
            cnt += 1

        self.assertEqual(cnt, exp_cnt)

    def test_gen_loopblocking_mp(self):
        ''' gen_loopblocking multiprocessing. '''

        cnt1 = 0
        for _ in loop_blocking.gen_loopblocking(
                self.nld['BASE'], self.resource['LG'], self.cost, 1,
                self.options['BASE']):
            cnt1 += 1

        cnt8 = 0
        for _ in loop_blocking.gen_loopblocking(
                self.nld['BASE'], self.resource['LG'], self.cost, 1,
                self.options['MP']):
            cnt8 += 1

        self.assertEqual(cnt1, cnt8)

    def test_gen_loopblocking_no_eqv(self):
        ''' gen_loopblocking no equivalent. '''

        acc_dict = {}

        for lbs in loop_blocking.gen_loopblocking(
                self.nld['BASE'], self.resource['LG'], self.cost, 1,
                self.options['BASE']):

            if not lbs.is_valid():
                continue

            sdict = lbs.get_scheme_dict(self.cost)

            # Make the keys hashable (list -> tuple).
            size = tuple(tuple(ss for ss in s) for s in sdict['size'])
            access = tuple(tuple(int(aa) for aa in a) for a in sdict['access'])
            keys = (size, access)

            self.assertNotIn(keys, acc_dict,
                             'test_gen_loopblocking_no_eqv: found equivalents. '
                             'keys: access {} size {}\n  {}\n  {}'
                             .format(access, size,
                                     sdict,
                                     acc_dict.get(keys)))
            acc_dict[keys] = sdict

    def test_gen_loopblocking_ntops(self):
        ''' gen_loopblocking ntops. '''

        tops = list(loop_blocking.gen_loopblocking(self.nld['BASE'],
                                                   self.resource['LG'],
                                                   self.cost, 1,
                                                   self.options['NTOPS']))

        cost_prev = -float('inf')

        for lbs in loop_blocking.gen_loopblocking(
                self.nld['BASE'], self.resource['LG'], self.cost, 1,
                self.options['BASE']):

            if not lbs.is_valid():
                continue

            cost_curr = lbs.get_cost(self.cost)
            self.assertLessEqual(cost_prev, cost_curr)
            cost_prev = cost_curr

            if tops:
                top_lbs = tops.pop(0)
                self.assertAlmostEqual(cost_curr, top_lbs.get_cost(self.cost))

    def test_gen_loopblocking_byp_sol(self):
        ''' gen_loopblocking using bypass solvers. '''

        cnt = 0

        for lbs in loop_blocking.gen_loopblocking(
                self.nld['BASE'], self.resource['BASE'], self.cost, 1,
                self.options['BYPSOL']):

            self.assertTrue(lbs.is_valid())

            cnt += 1

        self.assertLessEqual(cnt, 8)

