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

from . import data_category_enum as de
from . import loop_blocking_solver
from . import loop_enum as le
from .. import util
from .data_dim_loops import DataDimLoops
from .loop_blocking_scheme import LoopBlockingScheme

'''
Loop blocking optimization.

Include loop blocking and reordering.

For our problem, only deal with nifm, nofm, and batch loops.
'''

def skip_conv(bl_ts, bl_ords):
    '''
    Skip the given loop blocking scheme for CONV layer, if it has regularized
    equivalent, or it is suboptimal.

    Equivalence of loop blocking schemes:

    - changing the position of a trivial loop (with blocking factor 1) makes no
      difference to the access pattern.
    - reorder non-innermost non-trivial loops has no effect on reuse, although
      the access pattern changes.

    Therefore a scheme is regularized if:

    - all the trivial loops (with blocking factor 1) are at the outermost of
      this level, and are in order, i.e., smaller LoopEnum at inner.
    - the non-innermost non-trivial loops are in order, i.e., smaller LoopEnum
      at inner.

    A scheme is suboptimal if the closest innermost non-trivial loop of an
    outer level (skipping the levels with all trivial loops) is the same type
    (i.e., has the same LoopEnum value) as one of the non-innermost non-trivial
    loops of this level. For the last (innermost) level, all non-trivial loops
    should be considered, i.e., no innermost non-trivial loop.

    This is because an equivalent scheme can reorder the non-innermost loops to
    put the one loop adjacent to the outer-level innermost loop. Then this loop
    can be merged to the outer level, which results in the same access pattern
    but has smaller data size for this level.
    '''

    outer_level_innermost_nt_loop = None

    for t_, ord_ in itertools.izip_longest(bl_ts, bl_ords, fillvalue=None):

        # Non-trivial loops.
        nt_loops = [lpe for lpe in range(le.NUM) if t_[lpe] > 1]

        # Innermost non-trivial loops.
        try:
            innermost_nt_loop = min(nt_loops, key=lambda lpe, o=ord_: o[lpe])
        except (ValueError, TypeError):
            # All trivial loops, or order is None type (last level).
            innermost_nt_loop = None

        # Scheme is suboptimal if the outer-level innermost non-trivial loop is
        # a non-innermost non-trivial loops at this level.
        if outer_level_innermost_nt_loop != innermost_nt_loop \
                and outer_level_innermost_nt_loop in nt_loops:
            return True
        if innermost_nt_loop is not None:
            outer_level_innermost_nt_loop = innermost_nt_loop

        if ord_:
            # Order the LoopEnum values, from innermost to outermost.
            # The sort key is a three-tuple:
            # - innermost non-trivial loop should be kept at the innermost.
            # - non-trivial loops should be inside trivial loops.
            # - within each part, order by LoopEnum value.
            lp_ord = sorted(range(le.NUM),
                            key=lambda lpe, inl=innermost_nt_loop, nls=nt_loops:
                            (lpe != inl, lpe not in nls, lpe))

            if any(lp_ord[ord_[lpe]] != lpe for lpe in range(le.NUM)):
                return True

    return False


def _is_conv_loops(nested_loop_desc):
    return (nested_loop_desc.data_loops[de.FIL] == DataDimLoops(le.IFM, le.OFM)
            and nested_loop_desc.data_loops[de.IFM] \
                    == DataDimLoops(le.IFM, le.BAT)
            and nested_loop_desc.data_loops[de.OFM] \
                    == DataDimLoops(le.OFM, le.BAT))


def _gen_loopblocking_perprocess(
        nested_loop_desc, resource, cost, part_occ, options,
        gen_tifm, gen_tofm, gen_tbat, gen_ords):

    def _gen_bl_ts():
        '''
        Generator for blocking factors.

        Transpose LoopEnum-major to BL-major.
        '''
        gen_lp_ts = [None] * le.NUM
        gen_lp_ts[le.IFM] = gen_tifm
        gen_lp_ts[le.OFM] = gen_tofm
        gen_lp_ts[le.BAT] = gen_tbat
        for lp_ts in itertools.product(*gen_lp_ts):
            bl_ts = tuple(zip(*lp_ts))
            yield bl_ts

    def _sweep():
        ''' Sweep all. '''
        is_conv_loops = _is_conv_loops(nested_loop_desc)
        for bl_ts, bl_ords in itertools.product(_gen_bl_ts(), gen_ords):
            if is_conv_loops and skip_conv(bl_ts, bl_ords):
                continue
            lbs = LoopBlockingScheme(
                nested_loop_desc, bl_ts, bl_ords, resource, part_occ, options)
            yield lbs

    return heapq.nsmallest(options.ntops, _sweep(),
                           key=lambda lbs: lbs.get_cost(cost))


def gen_loopblocking(nested_loop_desc, resource, cost, part_occ, options):
    '''
    Generator for loop blocking.
    '''

    # Solver only works for CONV layer.
    if options.sw_solve_loopblocking and _is_conv_loops(nested_loop_desc):
        gen = loop_blocking_solver.gen_loopblocking_gbuf_reside

        for bl_ts, bl_ords in gen(nested_loop_desc, resource, options):
            lbs = LoopBlockingScheme(nested_loop_desc, bl_ts, bl_ords,
                                     resource, part_occ, options)
            yield lbs
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

    # Exhaustive generators.
    gen_tifm = util.factorize(nested_loop_desc.loopcnt[le.IFM], 3)
    gen_tofm = util.factorize(nested_loop_desc.loopcnt[le.OFM], 3)
    gen_tbat = util.factorize(nested_loop_desc.loopcnt[le.BAT], 3)
    gen_ords = itertools.product(itertools.permutations(range(le.NUM)),
                                 itertools.permutations(range(le.NUM)))

    # Split the design space for multiprocessing.
    # Let each process factorize tbat and orders, which constantly have many
    # factors that can amortize the multiprocessing overhead.
    # Note that we must materialize them into lists, since generators cannot be
    # pickled. See
    # http://peadrop.com/blog/2009/12/29/why-you-cannot-pickle-generators/
    list_tbat = list(gen_tbat)
    list_ords = list(gen_ords)
    for tifm, tofm in itertools.product(gen_tifm, gen_tofm):
        r = apply_func(_gen_loopblocking_perprocess,
                       (nested_loop_desc, resource, cost, part_occ, options,
                        [tifm], [tofm], list_tbat, list_ords))
        results.append(r)

    for lbs in heapq.nsmallest(options.ntops, retrieve_func,
                               key=lambda lbs: lbs.get_cost(cost)):
        yield lbs

    if pool is not None:
        pool.close()
        pool.join()

