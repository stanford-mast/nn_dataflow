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

import heapq
import itertools
from multiprocessing import Pool

from . import DataCategoryEnum as de
from . import LoopBlockingSolver
from . import LoopEnum as le
from . import MemHierEnum as me
from . import Util
from .LoopBlockingScheme import LoopBlockingScheme

'''
Loop blocking optimization.

Include loop blocking and reordering.

For our problem, only deal with nifm, nofm, and batch loops.
'''

_DEBUG = False

def loop_index_generator(ts_x, orders_x):
    '''
    Given the loop blocking factors and the order, generate the loop indexes.

    The blocking factors `ts_x` and order `orders_x` of the current hierarchy
    are both indexed by LoopEnum.

    Return the indexes in the order of LoopEnum.
    '''

    # Reversed order, i.e., values increase from outer to inner, while keys are
    # LoopEnum.
    rev_order = [le.NUM - 1 - o for o in orders_x]

    # Blocking factors, ordered from outer to inner.
    ts_o2i = [0] * le.NUM
    for lpe in range(le.NUM):
        ts_o2i[rev_order[lpe]] = ts_x[lpe]

    for idx_o2i in itertools.product(*[xrange(t) for t in ts_o2i]):
        yield tuple(idx_o2i[rev_order[lpe]] for lpe in range(le.NUM))


def _verify_loopblockingscheme_access_model(lbs):
    '''
    Verify the access model of LoopBlockingScheme by actually simulating and
    generating the loops.
    '''

    if not lbs.is_valid():
        return

    tifm = lbs.ti
    tofm = lbs.to
    tbat = lbs.tb
    orders = lbs.orders

    tip1 = Util.prod(tifm[1:])
    top1 = Util.prod(tofm[1:])
    tbp1 = Util.prod(tbat[1:])

    tip2 = Util.prod(tifm[2:])
    top2 = Util.prod(tofm[2:])
    tbp2 = Util.prod(tbat[2:])

    # Buffered data ranges in the gbuf/regf.
    # E.g., for FIL, (i0, o0) means the range [i0*tip1, i0*tip1 + tip1) x
    # [o0*top1, o0*top1 + top1) is in gbuf. IFM uses (i0, b0), and OFM uses
    # (o0, b0).
    buf_ranges = {me.GBUF: [(-1, -1)] * de.NUM,
                  me.REGF: [(-1, -1)] * de.NUM}
    # Data accesses from gbuf/regf to upper level.
    up_lvl_acc = {me.GBUF: [0] * de.NUM,
                  me.REGF: [0] * de.NUM}

    def _replace(mhe, dce, rng, size):
        '''
        Replace buffered data range for `dce` to be `rng` at level `mhe`, and
        update data accesses.
        '''
        if rng != buf_ranges[mhe][dce]:
            buf_ranges[mhe][dce] = rng
            up_lvl_acc[mhe][dce] += size

    # GBUF level.
    for i0, o0, b0 in loop_index_generator((tifm[0], tofm[0], tbat[0]),
                                           orders[0]):

        _replace(me.GBUF, de.FIL, (i0, o0), tip1 * top1)
        _replace(me.GBUF, de.IFM, (i0, b0), tip1 * tbp1)
        _replace(me.GBUF, de.OFM, (o0, b0), top1 * tbp1)

        # REGF level.
        for i1, o1, b1 in loop_index_generator((tifm[1], tofm[1], tbat[1]),
                                               orders[1]):

            # Assertions, so that [i01*tip2, i01*tip2 + tip2) is contained by
            # [i0*tip1, i0*tip1 + tip1), etc..
            assert i1 < tifm[1] and o1 < tofm[1] and b1 < tbat[1]
            i01 = i0 * tifm[1] + i1
            o01 = o0 * tofm[1] + o1
            b01 = b0 * tbat[1] + b1

            _replace(me.REGF, de.FIL, (i01, o01), tip2 * top2)
            _replace(me.REGF, de.IFM, (i01, b01), tip2 * tbp2)
            _replace(me.REGF, de.OFM, (o01, b01), top2 * tbp2)

    # Verify GBUF level to upper DRAM level accesses.
    dram_acc = [a * ua for a, ua
                in zip(up_lvl_acc[me.GBUF], lbs.unit_access[me.DRAM])]
    if not all(Util.isclose(a, b, rel_tol=1e-3, abs_tol=1.5)
               for a, b in zip(dram_acc, lbs.access[me.DRAM])):
        raise RuntimeError('LoopBlocking: verification failed for accesses to '
                           'DRAM for loop blocking scheme {}, {}, {}, {}: '
                           '{} vs. {}.'.format(tifm, tofm, tbat, orders,
                                               dram_acc, lbs.access[me.DRAM]))

    # Verify REGF level to upper GBUF level accesses.
    gbuf_acc = [a * ua * s for a, ua, s
                in zip(up_lvl_acc[me.REGF], lbs.unit_access[me.GBUF],
                       lbs.stored_in_gbuf)]
    if not all(Util.isclose(a, b, rel_tol=1e-3, abs_tol=1.5)
               for a, b in zip(gbuf_acc, lbs.access[me.GBUF])):
        raise RuntimeError('LoopBlocking: verification failed for accesses to '
                           'GBUF for loop blocking scheme {}, {}, {}, {}: '
                           '{} vs. {}.'.format(tifm, tofm, tbat, orders,
                                               gbuf_acc, lbs.access[me.GBUF]))


def _make_loopblockingscheme(nested_loop_desc, tifm, tofm, tbat, orders,
                             resource, part_occ, options):
    lbs = LoopBlockingScheme(nested_loop_desc, tifm, tofm, tbat, orders,
                             resource, options)
    lbs.set_partition_occupation(part_occ)
    return lbs


def _skip_ti_to_tb_orders(tifm, tofm, tbat, orders):
    '''
    Skip the given loop blocking scheme if:

    - trivial loops with blocking factor 1 are not all at the top.
    - the LP values of the outer two loops in each level are not in order,
      since the order of the outer two loops does not matter.
    - the innermost and outermost non-trivial loops of adjacent levels are the
      same, which is equal to merge into one loop at the outer level.
    '''

    outer_level_innermost_nt_loop = None

    for idx, mhe in enumerate([me.GBUF, me.REGF]):
        ord_ = orders[mhe]

        # Non-trivial loops.
        nt_loop_list = tuple(lpe for lpe, t in [(le.IFM, tifm[idx]),
                                                (le.OFM, tofm[idx]),
                                                (le.BAT, tbat[idx])] if t > 1)
        nt_loop_num = len(nt_loop_list)
        if not all(ord_[lpe] < nt_loop_num for lpe in nt_loop_list):
            return True

        # Outer two loops. Only allow the larger LoopEnum at the outermost.
        if nt_loop_num == le.NUM and (ord_[le.BAT] == 1 or ord_[le.IFM] == 2):
            return True

        # Outermost loop should not equal to the innermost loop of the outer
        # level.
        if nt_loop_num > 1:
            outermost_nt_loop = ord_.index(nt_loop_num - 1)
            if outermost_nt_loop == outer_level_innermost_nt_loop:
                return True
            outer_level_innermost_nt_loop = ord_.index(0)

    return False


def _loopblocking_iter_ti_to(nested_loop_desc, tbat, orders, resource, cost,
                             part_occ, options):
    def _sweep():
        for ti, to in itertools.product(
                Util.factorize(nested_loop_desc.loopcnt_ifm, 3),
                Util.factorize(nested_loop_desc.loopcnt_ofm, 3)):
            if (not _DEBUG) and _skip_ti_to_tb_orders(ti, to, tbat, orders):
                continue
            lbs = _make_loopblockingscheme(nested_loop_desc, ti, to, tbat,
                                           orders, resource, part_occ, options)
            if _DEBUG:
                _verify_loopblockingscheme_access_model(lbs)
            yield lbs

    return heapq.nsmallest(options.ntops, _sweep(),
                           key=lambda lbs: lbs.get_cost(cost))


def gen_loopblocking(nested_loop_desc, resource, cost, part_occ, options):
    '''
    Generator for loop blocking.
    '''

    if options.sw_solve_loopblocking:
        gen = LoopBlockingSolver.gen_loopblocking_gbuf_regf

        for ti, to, tb, orders in gen(nested_loop_desc, resource, options):
            yield _make_loopblockingscheme(nested_loop_desc, ti, to, tb,
                                           orders, resource, part_occ,
                                           options)
        return

    ## Exhaustive search.

    results = []

    def retrieve_result():
        ''' Retrieve results from multiprocessing.Pool. '''
        for r in results:
            for t in r.get(timeout=3600):
                yield t

    def retrieve_result_st():
        ''' Retrieve results from single-process processing. '''
        for r in results:
            for t in r:
                yield t

    if options.nprocesses > 1:
        pool = Pool(processes=options.nprocesses)
        apply_func = pool.apply_async
        retrieve_func = retrieve_result()
    else:
        pool = None
        apply_func = apply
        retrieve_func = retrieve_result_st()

    # Split the design space iteration for multiprocessing: make tb and orders
    # inter-process, and ti and to intra-process.
    for tb, orders in itertools.product(
            Util.factorize(nested_loop_desc.loopcnt_bat, 3),
            itertools.product(
                [None], itertools.permutations(range(le.NUM)),
                [None], itertools.permutations(range(le.NUM)))):
        r = apply_func(_loopblocking_iter_ti_to,
                       (nested_loop_desc, tb, orders, resource, cost, part_occ,
                        options))
        results.append(r)

    for lbs in heapq.nsmallest(options.ntops, retrieve_func,
                               key=lambda lbs: lbs.get_cost(cost)):
        yield lbs

    if pool is not None:
        pool.close()
        pool.join()

