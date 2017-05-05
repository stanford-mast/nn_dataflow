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

from collections import OrderedDict

from . import DataCategoryEnum as de
from . import MemHierEnum as me
from . import Util

class LoopBlockingScheme(object):
    '''
    Loop blocking scheme.

    Consider the loops of ifmap, ofmap, and batching.
    '''
    # pylint: disable=too-many-instance-attributes

    class BL(object):  # pylint: disable=too-few-public-methods
        '''
        Blocking-level enum. Only used locally.
        '''
        GBUF = 0
        REGF = 1
        NUM = 2

    def __init__(self, nested_loop_desc, tifm, tofm, tbat, orders,
                 resource, options):
        '''
        Given tiling factors `ti`, `to`, and `tb` for ifm, ofm and batching,
        and the loop `orders` of each tiling level, construct the loop blocking
        scheme.

        `orders` should be indexed by MemHierEnum, and only GBUF and REGF
        entries are valid. Each entry is a ordered tuple of IFM and OFM.
        Smaller index corresponds to inner loop. Batching loop order should
        never in between IFM and OFM, so we can enforce it to the outermost
        level for all memory hierarchy (innermost can be viewed as the
        outermost of the inner next hierarchy). So nested loop order is: tb[0],
        ti[0]/to[0], tb[1], ti[1]/to[1], tb[2], ti[2]/to[2]
        '''

        # pylint: disable=invalid-name
        BL = self.BL

        self.ti = tuple(tifm)
        self.to = tuple(tofm)
        self.tb = tuple(tbat)
        self.orders = orders

        self.tip = Util.prod(self.ti)
        self.top = Util.prod(self.to)
        self.tbp = Util.prod(self.tb)

        # Check lengths and values.
        assert len(self.ti) == BL.NUM + 1, 'LoopBlocking: wrong length for ti.'
        assert len(self.to) == BL.NUM + 1, 'LoopBlocking: wrong length for to.'
        assert len(self.tb) == BL.NUM + 1, 'LoopBlocking: wrong length for tb.'

        assert self.tip >= nested_loop_desc.loopcnt_ifm, \
                'LoopBlocking: invalid blocking for ifm: {}'.format(self.ti)
        assert self.top >= nested_loop_desc.loopcnt_ofm, \
                'LoopBlocking: invalid blocking for ofm: {}'.format(self.to)
        assert self.tbp >= nested_loop_desc.loopcnt_bat, \
                'LoopBlocking: invalid blocking for bat: {}'.format(self.tb)

        # Buffer data size for one unit.
        self.unit_size = [tuple() for _ in range(BL.NUM)]
        self.unit_size[BL.GBUF] = nested_loop_desc.usize_gbuf
        self.unit_size[BL.REGF] = nested_loop_desc.usize_regf

        # Buffer data unit counts.
        self.unit_cnt = [[0] * de.NUM for _ in range(BL.NUM)]
        for bl in range(BL.NUM):
            self.unit_cnt[bl][de.FIL] = \
                    Util.prod(self.ti[bl+1:]) * Util.prod(self.to[bl+1:])
            self.unit_cnt[bl][de.IFM] = \
                    Util.prod(self.ti[bl+1:]) * Util.prod(self.tb[bl+1:])
            self.unit_cnt[bl][de.OFM] = \
                    Util.prod(self.to[bl+1:]) * Util.prod(self.tb[bl+1:])

        # Whether reside in gbuf.
        self.stored_in_gbuf = [not options.sw_gbuf_bypass[dce]
                               for dce in range(de.NUM)]
        # Until now attribute `stored_in_gbuf` is conservative, i.e., assuming
        # all are False (bypassed) unless disabled.
        # They can be changed from False to True later, but never from True to
        # False.

        # Conservatively check size.
        if self.data_size(BL.REGF) > resource.size_regf \
                or self.data_size(BL.GBUF) > resource.size_gbuf:
            self.valid = False
            return
        else:
            self.valid = True

        # Data Reuse calculation.
        # Base reuse.
        self.reuse = [[0] * de.NUM for _ in range(BL.NUM)]
        for bl in range(BL.NUM):
            self.reuse[bl][de.FIL] = Util.prod(self.tb[bl+1:])
            self.reuse[bl][de.IFM] = Util.prod(self.to[bl+1:])
            self.reuse[bl][de.OFM] = Util.prod(self.ti[bl+1:])

        # Adjusted reuse based on loop orders, bypass, etc..
        self._adjust_reuse(self.reuse[BL.REGF], BL.REGF, self.orders[me.REGF],
                           [BL.GBUF], [self.orders[me.GBUF]])
        self._adjust_reuse(self.reuse[BL.GBUF], BL.GBUF, self.orders[me.GBUF],
                           [], [])

        # Now with the reuse, we can calculate the actual `stored_in_gbuf`
        # values.
        # Only store in gbuf if having reuse.
        for dce in range(de.NUM):
            # Skip enforced stored in gbuf.
            if self.stored_in_gbuf[dce]:
                continue
            assert options.sw_gbuf_bypass[dce]

            if self.reuse[BL.GBUF][dce] > self.reuse[BL.REGF][dce]:
                self.stored_in_gbuf[dce] = True

        # Recheck size.
        if self.data_size(BL.REGF) > resource.size_regf \
                or self.data_size(BL.GBUF) > resource.size_gbuf:
            self.valid = False
            return

        # Record unit access.
        self.unit_access = nested_loop_desc.unit_access

        # Misc.
        self.lcnt = self.tip * self.top * self.tbp
        self.ops = nested_loop_desc.unit_ops * self.lcnt
        self.time = nested_loop_desc.unit_time * self.lcnt

        self.part_occ = 1.  # set later.

        # Access.
        self._calc_access()

    def is_valid(self):
        '''
        Whether is a valid scheme.
        '''
        return self.valid

    def data_size(self, blvl, dce=None):
        '''
        Data sizes at the given blocking level.
        '''
        if dce is None:
            return sum(self.data_size(blvl, dce) for dce in range(de.NUM))

        size = self.unit_cnt[blvl][dce] * self.unit_size[blvl][dce]
        if blvl == self.BL.GBUF:
            size *= 1 if self.stored_in_gbuf[dce] else 0

        return size

    def set_partition_occupation(self, part_occ):
        ''' Set and scale by the given partitioning occupation. '''
        self.part_occ = part_occ
        self.scale_by_occupation(part_occ)

    def scale_by_occupation(self, occupation):
        '''
        Scale the computation and regf access by the given occupation.

        Other accesses are not affected.
        '''
        self.ops *= occupation
        self.access[me.REGF] = [a * occupation for a in self.access[me.REGF]]

    def get_access(self):
        '''
        Get number of accesses of each data category to each hierarchy.

        Access is a two-dimensional list, first indexed by MemHierEnum, then
        indexed by DataCategoryEnum.
        '''
        return self.access if self.is_valid() else None

    def get_fetches(self):
        '''
        Get number of top-level-hierarchy fetches of each data category.
        '''
        return self.fetches if self.is_valid() else None

    def get_cost(self, cost):
        '''
        Get the total cost of loop blocking.
        '''
        if not self.is_valid():
            return float('inf')

        c = 0

        c += self.ops * cost.mac_op

        access_total = [sum(acc) for acc in self.access]
        c += sum(mc * ma for mc, ma in zip(cost.mem_hier, access_total))

        c += self.time * cost.unit_static

        return c

    def get_scheme_dict(self):
        '''
        Get an OrderedDict of scheme summary.
        '''
        if not self.is_valid():
            return None

        size = [[self.data_size(bl, dce) for dce in range(de.NUM)]
                for bl in range(self.BL.NUM)]

        return OrderedDict([('ops', self.ops),
                            ('time', self.time),
                            ('access', self.access),
                            ('fetches', self.fetches),
                            ('size', size),
                            ('unit_size', self.unit_size),
                            ('unit_cnt', self.unit_cnt),
                            ('part_occ', self.part_occ),
                            ('ti', tuple(self.ti)),
                            ('to', tuple(self.to)),
                            ('tb', tuple(self.tb)),
                            ('orders', self.orders)])

    def _adjust_reuse(self, reuse_, bl_cur, order_cur, bls_outer, orders_outer):
        '''
        Adjust the data reuse based on special loop structures.

        reuse_ is the reuse numbers for a specific level, e.g., reuse[BL.REGF].

        This function is recursive as we need to look at the outer levels.
        '''
        if self.ti[bl_cur] != 1 and self.to[bl_cur] != 1:
            if order_cur.index(de.IFM) < order_cur.index(de.OFM):
                # Loop ifm inside loop ofm.
                # ofm also reused across current-level ifms.
                reuse_[de.OFM] *= self.ti[bl_cur]
            else:
                # Loop ifm outside loop ofm.
                # ifm also reused across current-level ofms.
                reuse_[de.IFM] *= self.to[bl_cur]
        elif self.ti[bl_cur] == 1 and self.to[bl_cur] != 1:
            # Current level does not change ifm, so ifm reuses ofms.
            reuse_[de.IFM] *= self.to[bl_cur]
        elif self.ti[bl_cur] != 1 and self.to[bl_cur] == 1:
            # Current level does not change ofm, so ofm reuses ifms.
            reuse_[de.OFM] *= self.ti[bl_cur]
        else:
            assert self.ti[bl_cur] == 1 and self.to[bl_cur] == 1
            # Current level loop counts are both 1 for ifms and ofms.
            # Effectively this level does not change the buffered data in the
            # inner level.
            # See the outer level.
            assert len(bls_outer) == len(orders_outer)
            if bls_outer:
                self._adjust_reuse(reuse_, bls_outer[0], orders_outer[0],
                                   bls_outer[1:], orders_outer[1:])

    def _calc_access(self):
        '''
        Calculate accesses to each hierarchy and the top-level fetches.
        '''
        # pylint: disable=invalid-name
        BL = self.BL

        # Accesses to each hierarchy.
        self.access = [[0] * de.NUM for _ in range(me.NUM)]

        self.access[me.REGF] = [v * self.lcnt
                                for v in self.unit_access[me.REGF]]

        self.access[me.ITCN] = [v * self.lcnt // r for v, r
                                in zip(self.unit_access[me.ITCN],
                                       self.reuse[BL.REGF])]

        self.access[me.GBUF] = [v * self.lcnt // r * s for v, r, s
                                in zip(self.unit_access[me.GBUF],
                                       self.reuse[BL.REGF],
                                       self.stored_in_gbuf)]

        self.access[me.DRAM] = [v * self.lcnt // r for v, r
                                in zip(self.unit_access[me.DRAM],
                                       self.reuse[BL.GBUF])]

        # Number of top-level (DRAM) fetches.
        self.fetches = [1] * de.NUM

        self.fetches[de.FIL] = self.tbp / self.reuse[BL.GBUF][de.FIL]

        if self.ti[BL.GBUF] != 1 \
                and self.orders[me.GBUF].index(de.IFM) \
                    < self.orders[me.GBUF].index(de.OFM):
            self.fetches[de.IFM] = self.to[BL.GBUF]
        assert self.fetches[de.IFM] * self.reuse[BL.GBUF][de.IFM] == self.top

        if self.to[BL.GBUF] != 1 \
                and self.orders[me.GBUF].index(de.OFM) \
                    < self.orders[me.GBUF].index(de.IFM):
            self.fetches[de.OFM] = self.ti[BL.GBUF]
        assert self.fetches[de.OFM] * self.reuse[BL.GBUF][de.OFM] == self.tip

