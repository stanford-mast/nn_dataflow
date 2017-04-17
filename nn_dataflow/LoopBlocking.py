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

'''
Loop blocking optimization.

Include loop blocking and reordering.

For our problem, only deal with nifm, nofm, and batch loops.
'''

import itertools
import numpy as np

from . import DataCategoryEnum as de
from . import MemHierEnum as me
from . import Util


class NestedLoopDesc(object):
    '''
    Naive 3-nested loop (nifm, nofm, batch) description.
    '''

    def __init__(self, lifms, lofms, lbats, ugbuf, uregf, uacc, uops, utime):
        self.loopcnts = (lifms, lofms, lbats)
        self.usize = (ugbuf, uregf)
        self.uacc = uacc
        self.uops = uops
        self.utime = utime

        for usz in self.usize:
            assert len(usz) == de.NUM

        assert len(self.uacc) == me.NUM
        for ua in range(me.NUM):
            assert len(self.uacc[ua]) == de.NUM

    def loopcnt_ifm(self):
        ''' Get nifm loop count. '''
        return self.loopcnts[0]

    def loopcnt_ofm(self):
        ''' Get nofm loop count. '''
        return self.loopcnts[1]

    def loopcnt_bat(self):
        ''' Get batch loop count. '''
        return self.loopcnts[2]

    def usize_gbuf(self, dce=None):
        '''
        Get occupied gbuf size of one innermost loop by data category `dce`.

        If None, return entire list of occupied gbuf sizes for all categories.
        '''
        return self.usize[0][dce] if dce is not None else self.usize[0]

    def usize_regf(self, dce=None):
        '''
        Get occupied regf size of one innermost loop by data category `dce`.

        If None, return entire list of occupied regf sizes for all categories.
        '''
        return self.usize[1][dce] if dce is not None else self.usize[1]

    def unit_access(self, mhe=None, dce=None):
        '''
        Get number of accesses of one innermost loop by memory hierarchy `mhe`
        of data category `dce`.

        If None, return entire list of accesses for the entire hierarchy.
        '''
        try:
            return self.uacc[mhe][dce]
        except (TypeError, IndexError):
            try:
                return self.uacc[mhe]
            except (TypeError, IndexError):
                return self.uacc

    def unit_num_ops(self):
        ''' Get number of ops of one innermost loop. '''
        return self.uops

    def unit_time(self):
        ''' Get execution time of one innermost loop. '''
        return self.utime

    def __str__(self):
        ''' Print. '''
        str_ = 'loopcnts={}'.format(self.loopcnts)
        str_ += ', usize={}'.format(self.usize)
        str_ += ', uacc={}'.format(self.uacc)
        str_ += ', uops={}'.format(self.uops)
        str_ += ', utime={}'.format(self.utime)
        return str_


def cost_loopblocking_gbuf_regf(tifm, tofm, tbat, orders,
                                resource, cost, nested_loop_desc, options):
    '''
    Given 2-tiled (length-3) `ti`, `to`, and `tb` for ifm, ofm and batching,
    and the loop `orders` of each tiling level, return the cost after loop
    blocking and the blocking parameters as a tuple (cost_loop, dict_loop).

    `orders` should be indexed by MemHierEnum, and only GBUF and REGF entries
    are valid. Each entry is a ordered tuple of IFM and OFM. Smaller index
    corresponds to inner loop. Batching loop order should never in between IFM
    and OFM, so we can enforce it to the outermost level for all memory
    hierarchy (innermost can be viewed as the outermost of the inner next
    hierarchy). So nested loop order is: tb[0], ti[0]/to[0], tb[1],
    ti[1]/to[1], tb[2], ti[2]/to[2]

    '''

    ## Input check.

    # Translate tx to np.ndarray, for better perf.
    ti = np.array(tifm)
    to = np.array(tofm)
    tb = np.array(tbat)

    tip = np.prod(ti)
    top = np.prod(to)
    tbp = np.prod(tb)

    # Check lengths and values.
    if ti.size != 3:
        raise ValueError('LoopBlocking: wrong length for ti.')
    if to.size != 3:
        raise ValueError('LoopBlocking: wrong length for to.')
    if tb.size != 3:
        raise ValueError('LoopBlocking: wrong length for tb.')

    class BL(object):  # pylint: disable=too-few-public-methods
        '''
        Blocking-level enum. Only used locally.
        '''
        GBUF = 0
        REGF = 1
        NUM = 2

    if tip < nested_loop_desc.loopcnt_ifm():
        raise ValueError('LoopBlocking: invalid blocking for ifm: {}'
                         .format(ti))
    if top < nested_loop_desc.loopcnt_ofm():
        raise ValueError('LoopBlocking: invalid blocking for ofm: {}'
                         .format(to))
    if tbp < nested_loop_desc.loopcnt_bat():
        raise ValueError('LoopBlocking: invalid blocking for bat: {}'
                         .format(tb))

    ## Buffer data sizes in unit counts.

    cnt_units = [None for _ in range(BL.NUM)]
    for bl in range(BL.NUM):
        cnt_units[bl] = [0] * de.NUM
        cnt_units[bl][de.FIL] = np.prod(ti[bl+1:]) * np.prod(to[bl+1:])
        cnt_units[bl][de.IFM] = np.prod(ti[bl+1:]) * np.prod(tb[bl+1:])
        cnt_units[bl][de.OFM] = np.prod(to[bl+1:]) * np.prod(tb[bl+1:])

    ## Num ops, time, etc.

    lcnt_total = tip * top * tbp

    ops_total = nested_loop_desc.unit_num_ops() * lcnt_total

    time_total = nested_loop_desc.unit_time() * lcnt_total

    ## Basic size and reuse.

    assert BL.GBUF == 0
    assert BL.REGF == 1
    unit_size = [[x for x in nested_loop_desc.usize_gbuf()],
                 [x for x in nested_loop_desc.usize_regf()]]
    reuse = [None for _ in range(BL.NUM)]
    for bl in range(BL.NUM):
        reuse[bl] = [0] * de.NUM
        reuse[bl][de.FIL] = np.prod(tb[bl+1:])
        reuse[bl][de.IFM] = np.prod(to[bl+1:])
        reuse[bl][de.OFM] = np.prod(ti[bl+1:])

    ## Adjusted size and reuse based on loop orders, bypass, etc..

    size = [None] * BL.NUM

    def adjust_reuse(reuse_, bl_cur, order_cur, bls_outer, orders_outer):
        '''
        Adjust the data reuse based on special loop structures.

        reuse_ is the reuse numbers for a specific level, e.g., reuse[BL.REGF].

        This function is recursive as we need to look at the outer levels.
        '''
        if ti[bl_cur] != 1 and to[bl_cur] != 1:
            if order_cur.index(de.IFM) < order_cur.index(de.OFM):
                # Loop ifm inside loop ofm.
                # ofm also reused across current-level ifms.
                reuse_[de.OFM] *= ti[bl_cur]
            else:
                # Loop ifm outside loop ofm.
                # ifm also reused across current-level ofms.
                reuse_[de.IFM] *= to[bl_cur]
        elif ti[bl_cur] == 1 and to[bl_cur] != 1:
            # Current level does not change ifm, so ifm reuses ofms.
            reuse_[de.IFM] *= to[bl_cur]
        elif ti[bl_cur] != 1 and to[bl_cur] == 1:
            # Current level does not change ofm, so ofm reuses ifms.
            reuse_[de.OFM] *= ti[bl_cur]
        else:
            assert ti[bl_cur] == 1 and to[bl_cur] == 1
            # Current level loop counts are both 1 for ifms and ofms.
            # Effectively this level does not change the buffered data in the
            # inner level.
            # See the outer level.
            assert len(bls_outer) == len(orders_outer)
            if len(bls_outer) > 0:
                adjust_reuse(reuse_, bls_outer[0], orders_outer[0],
                             bls_outer[1:], orders_outer[1:])

    # regf.
    adjust_reuse(reuse[BL.REGF], BL.REGF, orders[me.REGF],
                 [BL.GBUF], [orders[me.GBUF]])

    size[BL.REGF] = [np.prod(tuple_) for tuple_ in zip(unit_size[BL.REGF],
                                                       cnt_units[BL.REGF])]
    if sum(size[BL.REGF]) > resource.size_regf:
        return (float('inf'), None)

    # gbuf.
    adjust_reuse(reuse[BL.GBUF], BL.GBUF, orders[me.GBUF], [], [])

    stored_in_gbuf = [1] * de.NUM
    # Only store in gbuf if having reuse.
    for deum in range(de.NUM):
        stored_in_gbuf[deum] = 1 if not options.allow_gbuf_bypass[deum] \
                or reuse[BL.GBUF][deum] > reuse[BL.REGF][deum] \
                else 0

    size[BL.GBUF] = [np.prod(tuple_) for tuple_ in zip(unit_size[BL.GBUF],
                                                       cnt_units[BL.GBUF],
                                                       stored_in_gbuf)]
    if sum(size[BL.GBUF]) > resource.size_gbuf:
        return (float('inf'), None)

    ## Access.

    access = [0] * me.NUM

    access[me.REGF] = [v * lcnt_total for v
                       in nested_loop_desc.unit_access(me.REGF)]

    access[me.ITCN] = [v * lcnt_total // r for v, r
                       in zip(nested_loop_desc.unit_access(me.ITCN),
                              reuse[BL.REGF])]

    access[me.GBUF] = [v * lcnt_total // r * s for v, r, s
                       in zip(nested_loop_desc.unit_access(me.GBUF),
                              reuse[BL.REGF],
                              stored_in_gbuf)]

    access[me.DRAM] = [v * lcnt_total // r for v, r
                       in zip(nested_loop_desc.unit_access(me.DRAM),
                              reuse[BL.GBUF])]

    ## Cost.

    access_total = [sum(a) for a in access]
    cost_loop = np.dot(cost.mem_hier, access_total) \
            + ops_total * cost.mac_op \
            + time_total * cost.unit_static

    dict_loop = {'ops': ops_total,
                 'time': time_total,
                 'access': access,
                 'size': size,
                 'unit_size': unit_size,
                 'ti': tuple(ti),
                 'to': tuple(to),
                 'tb': tuple(tb),
                 'orders': orders}

    return (cost_loop, dict_loop)


def gen_loopblocking_gbuf_regf(resource, cost, nested_loop_desc, options):
    '''
    Generator for loop blocking schemes.
    '''
    for ti, to, tb, orders in itertools.product( \
            Util.factorize(nested_loop_desc.loopcnt_ifm(), 3),
            Util.factorize(nested_loop_desc.loopcnt_ofm(), 3),
            Util.factorize(nested_loop_desc.loopcnt_bat(), 3),
            itertools.product([None],
                              itertools.permutations((de.IFM, de.OFM)),
                              [None],
                              itertools.permutations((de.IFM, de.OFM)))):
        yield cost_loopblocking_gbuf_regf(ti, to, tb, orders,
                                          resource=resource, cost=cost,
                                          nested_loop_desc=nested_loop_desc,
                                          options=options)

