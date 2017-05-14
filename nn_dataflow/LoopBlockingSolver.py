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

import math
import itertools

from . import DataCategoryEnum as de
from . import Util

'''
Analytical solvers for loop blocking.
'''

def _solve_lpbl_iofmap_gbuf_reside(nested_loop_desc, resource, reside_dce):
    '''
    Given data category (ifm or ofm, according to `reside_dce` which is a
    DataCategortyEnum) is the only one in gbuf; the other data category and
    fil both bypass gbuf. Solve the analytical optimal loop blocking. Return
    ti, to, tb, and orders, same format as
    LoopBlocking.gen_loopblocking_gbuf_regf().

    Denote xfm to be the one bypassing, yfm to be the other one residing.

    Nested loop is:
    tb[0], ty[0], tx[0], (tb[1] = 1), (tx[1] = 1), ty[1], tb[2], tx[2]/ty[2].

    Note that ty[0] outside tx[0] means we stream the bypassing xfm multiple
    times (equal to ty[0]), each for one chunk of yfm in gbuf. Otherwise, yfm
    will be accessed multiple times (equal to tx[0]). Because ty[0] is gbuf
    chunk count and tx[0] is regf chunk count (tx[1] = 1), ty[0] is likely
    smaller and thus better.

    Also note that tx[1] outside ty[1]. This indicates that xfm which is
    streamed in bypassing gbuf is loaded into regf once, and then each part of
    yfm chunk in gbuf is loaded into regf. Reversing this order is wrong, as
    that means xfm streaming into regf multiple times for each gbuf-bypass
    streaming, which requires store in gbuf.

    Opt I.

    min accesses to DRAM =
        (Nx * sx * B) * fx * ty + (Ny * sy * B) * fy + (Nx * Ny * sf) * tb
    s.t.
        (Ny * sy / ty) * (B / tb) <= Sgbuf
        1 <= ty <= Ny
        1 <= tb <= B

    Nx, Ny, B are numbers of xfms, yfms, and batch size.
    sx, sy, sf are size of one xfm, yfm, fil.
    ty, tb are top-level tiling for yfm and batch.

    Opt II.

    min refetch yfm from gbuf =
        (Nx / nx)
    s.t.
        nx * sx * nb + ny * sy * nb + nx * ny * sf <= Sregf

    nx, ny are numbers of xfms, yfms in regf. nb is number of batches in regf.

    Solving Opt I and Opt II will give the results for ifmap or ofmap
    bypassing.
    '''

    dce_y = reside_dce
    if dce_y == de.OFM:
        dce_x = de.IFM
        nfmaps_x = nested_loop_desc.loopcnt_ifm
        nfmaps_y = nested_loop_desc.loopcnt_ofm
        facc_x = 1
        facc_y = 2
    elif dce_y == de.IFM:
        dce_x = de.OFM
        nfmaps_x = nested_loop_desc.loopcnt_ofm
        nfmaps_y = nested_loop_desc.loopcnt_ifm
        facc_x = 2
        facc_y = 1
    else:
        raise RuntimeError('LoopBlockingSolver: only allow ifmap or ofmap '
                           'to bypass.')

    nbats = nested_loop_desc.loopcnt_bat

    usize_gbuf_x = nested_loop_desc.usize_gbuf_of(dce_x)
    usize_gbuf_y = nested_loop_desc.usize_gbuf_of(dce_y)
    usize_gbuf_fil = nested_loop_desc.usize_gbuf_of(de.FIL) + 1e-6

    max_size_gbuf = resource.size_gbuf

    usize_regf_x = nested_loop_desc.usize_regf_of(dce_x)
    usize_regf_y = nested_loop_desc.usize_regf_of(dce_y)
    usize_regf_fil = nested_loop_desc.usize_regf_of(de.FIL) + 1e-6

    max_size_regf = resource.size_regf

    # Opt I problem.
    def goal(ty, tb):  # pylint: disable=invalid-name
        ''' Goal function. min goal(). '''
        return ((nfmaps_x * usize_gbuf_x * nbats) * facc_x * ty
                + (nfmaps_y * usize_gbuf_y * nbats) * facc_y
                + (nfmaps_x * nfmaps_y * usize_gbuf_fil) * tb)
    def constraints(ty, tb):  # pylint: disable=invalid-name
        ''' All constraints. s.t. constraints(). '''
        c1 = ((nfmaps_y * usize_gbuf_y / float(ty)) * (nbats / float(tb))
              < max_size_gbuf)
        return c1

    # Candidates of optimal ty, tb.
    ty_tb_cands = []

    # Analytical solution for min goal() s.t. constraints().
    ty_top = nfmaps_y * math.sqrt(float(usize_gbuf_fil * usize_gbuf_y)
                                  / (usize_gbuf_x * max_size_gbuf * facc_x))
    tb_top = nbats * math.sqrt(float(usize_gbuf_x * usize_gbuf_y * facc_x)
                               / (usize_gbuf_fil * max_size_gbuf))
    # Enforce to be a factor of total loop count.
    ty_top_adj = Util.closest_factor(nfmaps_y, ty_top)
    tb_top_adj = Util.closest_factor(nbats, tb_top)
    # Add to candidates.
    ty_tb_cands += itertools.product(ty_top_adj, tb_top_adj)

    # Boundary points.
    # When tb = B. Solve constraints().
    tb_bnd = nbats
    ty_bnd = nfmaps_y * usize_gbuf_y / float(max_size_gbuf)
    tb_bnd_adj = [tb_bnd]
    ty_bnd_adj = Util.closest_factor(nfmaps_y, ty_bnd)
    # Add to candidates.
    ty_tb_cands += itertools.product(ty_bnd_adj, tb_bnd_adj)
    # When tb = 1. Solve constraints().
    tb_bnd = 1
    ty_bnd = nfmaps_y * nbats * usize_gbuf_y / float(max_size_gbuf)
    tb_bnd_adj = [tb_bnd]
    ty_bnd_adj = Util.closest_factor(nfmaps_y, ty_bnd)
    # Add to candidates.
    ty_tb_cands += itertools.product(ty_bnd_adj, tb_bnd_adj)
    # When ty = Ny. Solve constraints().
    ty_bnd = nfmaps_y
    tb_bnd = nbats * usize_gbuf_y / float(max_size_gbuf)
    ty_bnd_adj = [ty_bnd]
    tb_bnd_adj = Util.closest_factor(nbats, tb_bnd)
    # Add to candidates.
    ty_tb_cands += itertools.product(ty_bnd_adj, tb_bnd_adj)
    # When ty = 1. Solve constraints().
    ty_bnd = 1
    tb_bnd = nfmaps_y * nbats * usize_gbuf_y / float(max_size_gbuf)
    ty_bnd_adj = [ty_bnd]
    tb_bnd_adj = Util.closest_factor(nbats, tb_bnd)
    # Add to candidates.
    ty_tb_cands += itertools.product(ty_bnd_adj, tb_bnd_adj)

    # Select best ty, tb from candidates.
    best_ty_tb = min([(goal(*ty_tb_), ) + ty_tb_ for ty_tb_ in ty_tb_cands
                      if constraints(*ty_tb_)])

    tb0 = best_ty_tb[2]
    # Because fil bypasses gbuf, tb[1] = 1.
    tb1 = 1
    tb2 = nbats / tb0 / tb1
    tb = (tb0, tb1, tb2)

    ty0 = best_ty_tb[1]
    # Opt II problem is trivial to solve, let ny = 1 and max nx
    ty2 = 1
    ty1 = nfmaps_y / ty0 / ty2
    ty = (ty0, ty1, ty2)

    tx_bottom = float(max_size_regf - usize_regf_y * ty[2] * tb[2]) \
                / (usize_regf_x * tb[2] + usize_regf_fil)
    tx2 = Util.closest_factor(nfmaps_x, tx_bottom)[0]
    # Because xfm bypasses gbuf, tx[1] = 1.
    tx1 = 1
    tx0 = nfmaps_x / tx1 / tx2
    tx = (tx0, tx1, tx2)

    # Compose return values.
    # For orders see docstring: at gbuf, b, y, x; at regf, b, x, y.
    if dce_x == de.IFM:
        ti = tx
        to = ty
        orders = (None, (0, 1, 2), None, (1, 0, 2))
    elif dce_x == de.OFM:
        ti = ty
        to = tx
        orders = (None, (1, 0, 2), None, (0, 1, 2))

    return ti, to, tb, orders


def _solve_lpbl_filter_gbuf_reside(nested_loop_desc, resource):
    '''
    The fil is the only one in gbuf; both ifm and ofm bypass gbuf. Solve the
    analytical optimal loop blocking. Return ti, to, tb, and orders, same
    format as LoopBlocking.gen_loopblocking_gbuf_regf().

    Denote xfm loop to be inside yfm loop at the outermost level.

    Nested loop is:
    (tb[0] = 1), ty[0], tx[0], tb[1], (ty[1] = 1), (tx[1] = 1), tb[2],
    tx[2]/ty[2].

    tb[0] = 1 is because both fmaps bypass gbuf, and we loop across all batches
    in the middle level tb[1].

    Opt I.

    min accesses to DRAM =
        (Nx * sx * B) * fx * ty + (Ny * sy * B) * fy + (Nx * Ny * sf)
    s.t.
        nx * sx * nb + ny * sy * nb + nx * ny * sf <= Sregf
        nx * ny * sf <= Sgbuf

    Each time we bring in nx xfms and ny yfms with batch size nb into regf.

    Solving Opt I will give the results for ifmap and ofmap both bypassing.
    '''

    goal_res = float('inf')

    for dce_y in [de.IFM, de.OFM]:
        if dce_y == de.OFM:
            dce_x = de.IFM
            nfmaps_x = nested_loop_desc.loopcnt_ifm
            nfmaps_y = nested_loop_desc.loopcnt_ofm
            facc_x = 1
            facc_y = 2
        elif dce_y == de.IFM:
            dce_x = de.OFM
            nfmaps_x = nested_loop_desc.loopcnt_ofm
            nfmaps_y = nested_loop_desc.loopcnt_ifm
            facc_x = 2
            facc_y = 1
        else:
            raise RuntimeError('LoopBlockingSolver: only allow ifmap or ofmap '
                               'to bypass.')

        nbats = nested_loop_desc.loopcnt_bat

        usize_gbuf_x = nested_loop_desc.usize_gbuf_of(dce_x)
        usize_gbuf_y = nested_loop_desc.usize_gbuf_of(dce_y)
        usize_gbuf_fil = nested_loop_desc.usize_gbuf_of(de.FIL) + 1e-6

        max_size_gbuf = resource.size_gbuf

        usize_regf_x = nested_loop_desc.usize_regf_of(dce_x)
        usize_regf_y = nested_loop_desc.usize_regf_of(dce_y)
        usize_regf_fil = nested_loop_desc.usize_regf_of(de.FIL) + 1e-6

        max_size_regf = resource.size_regf

        # To minimize Opt I, minimize ty, i.e., maximize ny.
        # Thus in constraints, set nx = 1 and nb = 1 to solve ny.
        ny_top_cand = min(float(max_size_regf - usize_regf_x) \
                          / (usize_regf_y + usize_regf_fil),
                          float(max_size_gbuf) / usize_gbuf_fil)
        # Pick max-no-larger factor to stay with constraint.
        ny_top = Util.closest_factor(nfmaps_y, ny_top_cand)[0]
        assert ny_top <= ny_top_cand
        ty_top = nfmaps_y / ny_top
        # Re-solve to maximize nb by exploiting the margine.
        nb_top_cand = float(max_size_regf - usize_regf_fil * ny_top)\
                      / (usize_regf_x + usize_regf_y * ny_top)
        # Pick max-no-larger factor to stay with constraint.
        nb_top = Util.closest_factor(nbats, nb_top_cand)[0]
        assert nb_top <= nb_top_cand

        # Goal value.
        goal_val = ((nfmaps_x * usize_gbuf_x * nbats) * facc_x * ty_top
                    + (nfmaps_y * usize_gbuf_y * nbats) * facc_y
                    + (nfmaps_x * nfmaps_y * usize_gbuf_fil))
        # Constraints.
        c1 = (1 * usize_regf_x * 1 + ny_top * usize_regf_y * 1
              + 1 * ny_top * usize_regf_fil <= max_size_regf)
        c2 = (1 * ny_top * usize_gbuf_fil <= max_size_gbuf)
        assert c1 and c2

        if goal_val < goal_res:

            # tb[0] = 1 due to docstring, tb[2] = nb_top.
            tb = (1, nbats / nb_top, nb_top)
            # ty[1] = 1 due to docstring, ty[0] = ty_top, ty[2] = ny_top.
            ty = (ty_top, 1, nfmaps_y / ty_top)
            # tx[1] = 1 due to docstring, tx[2] = nx = 1.
            tx = (nfmaps_x, 1, 1)

            # Compose return values.
            # For orders see docstring: at gbuf, b, y, x; at regf, b, y, x.
            if dce_x == de.IFM:
                ti = tx
                to = ty
                orders = (None, (0, 1, 2), None, (0, 1, 2))
            elif dce_x == de.OFM:
                ti = ty
                to = tx
                orders = (None, (1, 0, 2), None, (1, 0, 2))

    return ti, to, tb, orders


def gen_loopblocking_gbuf_regf(nested_loop_desc, resource, options):
    '''
    Generator for loop blocking schemes that are solved from iofmap gbuf bypass
    analytical models.
    '''
    reside_dce_list = []
    # reside_dce_list is a list of DataCategoryEnum, each element is a config
    # with only that data category in gbuf, i.e., the others are all bypassed.
    for reside_dce in range(de.NUM):
        if all([denum == reside_dce or options.sw_gbuf_bypass[denum]
                for denum in range(de.NUM)]):
            reside_dce_list.append(reside_dce)

    for reside_dce in reside_dce_list:
        if reside_dce == de.FIL:
            ti, to, tb, orders = _solve_lpbl_filter_gbuf_reside(
                nested_loop_desc, resource)
        else:
            assert reside_dce == de.IFM or reside_dce == de.OFM
            ti, to, tb, orders = _solve_lpbl_iofmap_gbuf_reside(
                nested_loop_desc, resource, reside_dce)
        yield ti, to, tb, orders

