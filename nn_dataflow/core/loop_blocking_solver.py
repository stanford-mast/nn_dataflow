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

import math

from . import data_category_enum as de
from . import loop_enum as le
from .. import util
from .layer import ConvLayer

'''
Analytical solvers for loop blocking.
'''

def _solve_gbuf_reside(nested_loop_desc, resource, reside_dce):
    '''
    Solve the analytical optimal loop blocking scheme, with the given data
    category `reside_dce` is the only one in GBUF; all the other data
    categories bypass GBUF.

    At the GBUF blocking level, the loops for the reside data category are at
    the outer, meaning it is only accessed once into GBUF. The others bypass
    GBUF and are streamed multiple times from DRAM to REGF.

    Let x, y, z be the three LoopEnum values, and x, y are for `reside_dce`,
    then the nested loop is:

    tx0/ty0, tz0, (tz1 = 1), tx1/ty1 (= 1, ?), tx2/ty2/tz2

    Note that tz1 = 1 is required since tz0 is the innermost of the outer level
    (otherwise tz1 can merge into tz0). The REGF level can only allow one
    non-trivial loop, so either tx1 or ty1 must also be 1.

    Opt I.

    min accesses to DRAM =
        (Nx * Ny * sgxy) * fxy + (Ny * Nz * sgyz) * fyz * tx0
        + (Nx * Nz * sgxz) * fxz * ty0
    s.t.
        1 <= tx0 <= Nx
        1 <= ty0 <= Ny
        (Nx // tx0) * (Ny // ty0) * sgxy <= Sgbuf
        min{(srxy + srxz) * (Nx // tx0) + sryz,
            (srxy + sryz) * (Ny // ty0) + srxz} <= Sregf

    Nx, Ny, Nz are the total loop factors.
    sgxy, sgyz, sgxz are the data unit sizes in GBUF.
    srxy, sryz, srxz are the data unit sizes in REGF.

    The last constraint is for the feasibility of REGF capacity. Note that tz2
    could be 1. If ty2 is minimized to 1 (so ty1 is not 1), tx1 must be 1;
    similarly, if tx2 is minimized to 1 (so tx1 is not 1), ty1 must be 1. At
    least one of these two cases must be feasible for REGF capacity.

    Although opt I is a convex optimization, we need to further require tx0 and
    ty0 to be factors of Nx and Ny, respectively. So we use exhaustive search
    to solve opt I.

    Opt II.

    min fetch to GBUF for `reside_dce` =
        1           if tx1 = ty1 = 1
        tz0         elsewise
    s.t.
        tx2 * ty2 * srxy + ty2 * tz2 * sryz + tx2 * tz2 * srxz <= Sregf

    If tx1 and ty1 could be 1, which means the reside data category could put
    all GBUF data Nx // tx0 and Ny // ty0 directly into REGF, then it is the
    optimal case.

    Otherwise, since tz1 = 1, min tz0 is equivalent to

    max tz2 =
        (Sregf - tx2 * ty2 * srxy) / (ty2 * sryz + tx2 * srxz)

    Special adjustment.

    The above model assumes tz0 is a non-trivial loop. If the final solution
    has tz0 = 1, the bypass data categories may not bypass. For example, if ty0
    is the innermost loop of the top level, data xz will have 1 fetch to DRAM,
    but ty0 fetch to GBUF. So we have to adjust the scheme by merging tx1 or
    ty1 into tx0 or ty0, and ensure it to be the inner loop at the top level.
    '''

    ldce = [reside_dce]  # xy, yz, xz
    llpe = []  # x, y, z
    lfacc = []  # xy, yz, xz

    if ldce[0] == de.FIL:
        llpe += [le.IFM, le.OFM, le.BAT]
        ldce += [de.OFM, de.IFM]
        lfacc += [1., 2., 1.]
    elif ldce[0] == de.IFM:
        llpe += [le.IFM, le.BAT, le.OFM]
        ldce += [de.OFM, de.FIL]
        lfacc += [1., 2., 1.]
    else:
        assert ldce[0] == de.OFM
        llpe += [le.OFM, le.BAT, le.IFM]
        ldce += [de.IFM, de.FIL]
        lfacc += [2., 1., 1.]

    lnum = [nested_loop_desc.loopcnt[lpe] for lpe in llpe]  # x, y, z
    lsgbuf = [nested_loop_desc.usize_gbuf_of(dce) for dce in ldce]  # xy, yz, xz
    lsregf = [nested_loop_desc.usize_regf_of(dce) for dce in ldce]  # xy, yz, xz

    size_gbuf, size_regf = resource.size_gbuf, resource.size_regf

    def goal_opt1(tx0, ty0):
        ''' Opt I goal function. min goal(). '''
        lnumloops = [lnum[0] * lnum[1], lnum[1] * lnum[2], lnum[0] * lnum[2]]
        ltloops = [1, tx0, ty0]
        return sum(util.prod(tpl) for tpl
                   in zip(lnumloops, lsgbuf, lfacc, ltloops))

    def constraints_opt1(tx0, ty0):
        ''' Opt I constraints. s.t. constraints(). '''
        if (lnum[0] // tx0) * (lnum[1] // ty0) * lsgbuf[0] > size_gbuf:
            return False
        if min(lnum[0] // tx0 * (lsregf[0] + lsregf[2]) + lsregf[1],
               lnum[1] // ty0 * (lsregf[0] + lsregf[1]) + lsregf[2]) \
                       > size_regf:
            return False
        return True

    # Exhaustive search for opt I.
    min_goal = float('inf')
    for tx0_, _ in util.factorize(lnum[0], 2):
        for ty0_, _ in util.factorize(lnum[1], 2):
            # Satisfy constraints.
            if not constraints_opt1(tx0_, ty0_):
                continue
            # Minimize goal.
            goal = goal_opt1(tx0_, ty0_)
            if goal < min_goal:
                min_goal = goal
                tx0, ty0 = tx0_, ty0_

    def goal_opt2(tx2, ty2):
        ''' Opt II goal function. max goal(). '''
        tz2 = (size_regf - tx2 * ty2 * lsregf[0]) * 1. \
                / (ty2 * lsregf[1] + tx2 * lsregf[2])
        if tz2 < 0:
            return -float('inf')
        tz2_adj = util.closest_factor(lnum[2], tz2)
        if tz2_adj[0] <= tz2:
            return tz2_adj[0]
        return -float('inf')

    # Try tx1 = ty1 = 1.
    tx2, ty2 = lnum[0] // tx0, lnum[1] // ty0
    tz2 = goal_opt2(tx2, ty2)

    if math.isinf(tz2):
        # Candidates of tx2, ty2.
        txy2_cands = [(1, lnum[1] // ty0), (lnum[0] // tx0, 1)]

        # Select.
        tx2, ty2 = max(txy2_cands, key=lambda txy2: goal_opt2(*txy2))
        tz2 = goal_opt2(tx2, ty2)

    assert not math.isinf(tz2)
    tz0 = lnum[2] // tz2
    tx1 = lnum[0] // tx0 // tx2
    ty1 = lnum[1] // ty0 // ty2

    # Loop orders.
    # Loop z is at the innermost of the top level. Do not care x, y.
    bl_ord_0 = [0] * le.NUM
    bl_ord_0[llpe[0]] = 2
    bl_ord_0[llpe[1]] = 1
    bl_ord_0[llpe[2]] = 0
    # The non-1 loop x or y is at the innermost of the middle level.
    bl_ord_1 = [0] * le.NUM
    bl_ord_1[llpe[0]] = 0 if tx1 > 1 else 1
    bl_ord_1[llpe[1]] = 1 if tx1 > 1 else 0
    bl_ord_1[llpe[2]] = 2

    # Special adjustment when tz0 = 1: merge tx1/ty1 into tx0/ty0.
    if tz0 == 1:
        tx0 *= tx1
        tx1 = 1
        ty0 *= ty1
        ty1 = 1
        # Also maintain the order.
        bl_ord_0 = bl_ord_1

    # Compose return values.
    lp_ts = [None] * le.NUM
    lp_ts[llpe[0]] = (tx0, tx1, tx2)
    lp_ts[llpe[1]] = (ty0, ty1, ty2)
    lp_ts[llpe[2]] = (tz0, 1, tz2)
    bl_ts = tuple(zip(*lp_ts))

    bl_ords = (tuple(bl_ord_0), tuple(bl_ord_1))

    return bl_ts, bl_ords


def gen_loopblocking_gbuf_reside(nested_loop_desc, resource, options):
    '''
    Generator for loop blocking schemes that are solved from gbuf reside
    analytical models.
    '''
    if nested_loop_desc.data_loops != ConvLayer.data_loops():
        raise ValueError('loop_blocking_solver: solver only applies to '
                         'CONV layer nested loops')

    reside_dce_list = []
    # reside_dce_list is a list of DataCategoryEnum, each element is a config
    # with only that data category in gbuf, i.e., the others are all bypassed.
    for reside_dce in range(de.NUM):
        if all(options.sw_gbuf_bypass[dce] for dce in range(de.NUM)
               if dce != reside_dce):
            reside_dce_list.append(reside_dce)

    for reside_dce in reside_dce_list:
        yield _solve_gbuf_reside(nested_loop_desc, resource, reside_dce)

