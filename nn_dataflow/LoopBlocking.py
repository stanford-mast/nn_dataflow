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
                lbs.verify_fetch()
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

