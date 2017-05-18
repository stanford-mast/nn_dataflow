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
from . import LoopEnum as le
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

        Nested loop order is:

        for ti[0]/to[0]/tb[0]
          // The data access order at this point (determined by the loop
          // order above) determines the access to DRAM.
          //
          // ------ boundary of DRAM and GBUF levels ------
          //
          // Data ranges below in this loop body are buffered in GBUF.
          for ti[1]/to[1]/tb[1]
            // The data access order at this point (determined by the loop
            // order above) determines the access to GBUF.
            //
            // ------ boundary of GBUF and REGF levels ------
            //
            // Data ranges below in this loop body are buffered in REGF.
            for ti[2]/to[2]/tb[2]

        `orders` indicates the order of ifm, ofm, bat loops at each level. Only
        GBUF and REGF entries are valid. It is indexed by MemHierEnum, and each
        entry is a 3-permutation of (0, 1, 2), which is indexed by LoopEnum and
        gives the position of the ifm, ofm, bat loops. Smaller number means
        inner loop.
        '''

        # pylint: disable=invalid-name
        BL = self.BL

        self.ti = tuple(tifm)
        self.to = tuple(tofm)
        self.tb = tuple(tbat)

        self.orders = [tuple() for _ in range(BL.NUM)]
        self.orders[BL.GBUF] = tuple(orders[me.GBUF])
        self.orders[BL.REGF] = tuple(orders[me.REGF])

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
        self._set_unit_cnt()

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

        # Data fetch calculation.
        self._set_fetch()

        # Now with the fetch times, we can calculate the actual
        # `stored_in_gbuf` values.
        # Only store in gbuf if having reuse.
        for dce in range(de.NUM):
            # Skip enforced stored in gbuf.
            if self.stored_in_gbuf[dce]:
                continue
            assert options.sw_gbuf_bypass[dce]

            if self.fetch[BL.GBUF][dce] < self.fetch[BL.REGF][dce]:
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
        if not self.is_valid():
            return
        self.part_occ = part_occ
        self._scale_by_occupation(part_occ)

    def _scale_by_occupation(self, occupation):
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

    def get_top_level_fetch(self):
        '''
        Get number of top-level-hierarchy fetches of each data category.
        '''
        return self.fetch[self.BL.GBUF] if self.is_valid() else None

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
                            ('fetch', self.fetch),
                            ('size', size),
                            ('unit_size', self.unit_size),
                            ('unit_cnt', self.unit_cnt),
                            ('part_occ', self.part_occ),
                            ('ti', tuple(self.ti)),
                            ('to', tuple(self.to)),
                            ('tb', tuple(self.tb)),
                            ('orders', self.orders)])

    def _set_unit_cnt(self):
        '''
        Set the buffered unit counts for all data categories at all blocking
        levels, based on the loop blocking factors and orders.

        General rules:
        - from the top of the current level, go down (inner) and multiply up
          all blocking factors of loops that are related to the data (e.g.,
          loop i and b for IFM).
        - the product is the buffered unit count.
        '''

        self.unit_cnt = []

        for bl in range(self.BL.NUM):
            # BL corresponds to the BL + 1 element in ti/to/tb.
            blp1 = bl + 1
            pblti = Util.prod(self.ti[blp1:])
            pblto = Util.prod(self.to[blp1:])
            pbltb = Util.prod(self.tb[blp1:])

            uc = [1] * de.NUM
            uc[de.FIL] = pblti * pblto
            uc[de.IFM] = pblti * pbltb
            uc[de.OFM] = pblto * pbltb

            self.unit_cnt.append(uc)

    def _set_fetch(self):
        '''
        Set the data fetch times for all data categories at all blocking
        levels, based on the loop blocking factors and orders.

        Fetch times considers the buffering. E.g., for IFM, there are `tip` *
        `tbp` units, which need to be fetched `top` times without buffering.
        With reuse due to buffering, the actual fetch times will be `top` /
        reuse.

        Fetch times at a level means the fetch to the upper level. E.g.,
        fetches at GBUF level access DRAM, and fetches at REGF level accesses
        GBUF.

        General rules:
        - from the top of the current level, go up (outer) until hitting a
          non-trivial (blocking factor > 1) loop that is related to the data
          category (e.g., loop i and b for IFM).
        - start from that loop, go up (outer) until the outermost, and multiply
          up all blocking factors of loops that are related to the data that
          will reuse this data, but are unrelated to this data (e.g., loop o
          for IFM).
        - the product is the fetch times.
        '''

        self.fetch = []

        # Have to go from outer levels to inner levels.
        assert self.BL.GBUF < self.BL.REGF
        for bl in range(self.BL.NUM):
            # BL corresponds to the BL + 1 element in ti/to/tb. But the outer
            # level is the BL element.

            # If the blocking factors of a data category are all 1's in the
            # current level, the current level does not change the data, and
            # the fetch times at this level is the same as the outer level.

            # Every data category has two related loops and one unrelated
            # loops. Only when the innermost non-trivial loop of the current
            # level is the unrelated loop, can the data reuse include the
            # current level blocking factor.

            # The innermost non-trivial loop.
            # If all loops are trivial, we will use the outer level reuse for
            # all data, so the loop is not used.
            innermost_nt_lp = self._innermost_nontrivial_loop(bl)

            fe = [0] * de.NUM

            if self.ti[bl] * self.to[bl] == 1:
                fe[de.FIL] = self.fetch[bl-1][de.FIL] if bl > 0 else 1
            else:
                bl_start = bl + (innermost_nt_lp != le.BAT)
                fe[de.FIL] = Util.prod(self.tb[:bl_start])

            if self.ti[bl] * self.tb[bl] == 1:
                fe[de.IFM] = self.fetch[bl-1][de.IFM] if bl > 0 else 1
            else:
                bl_start = bl + (innermost_nt_lp != le.OFM)
                fe[de.IFM] = Util.prod(self.to[:bl_start])

            if self.to[bl] * self.tb[bl] == 1:
                fe[de.OFM] = self.fetch[bl-1][de.OFM] if bl > 0 else 1
            else:
                bl_start = bl + (innermost_nt_lp != le.IFM)
                # See equation (4) in Chen, et al., ISCA'16.
                fe[de.OFM] = 2 * Util.prod(self.ti[:bl_start]) \
                        - Util.prod(self.ti[:max(0, bl_start-1)])

            self.fetch.append(fe)

    def _calc_access(self):
        '''
        Calculate accesses to each hierarchy.
        '''

        self.access = [[0] * de.NUM for _ in range(me.NUM)]

        total_units = [0] * de.NUM
        total_units[de.FIL] = self.tip * self.top
        total_units[de.IFM] = self.tip * self.tbp
        total_units[de.OFM] = self.top * self.tbp

        self.access[me.REGF] = [v * self.lcnt * t for v, t
                                in zip(self.unit_access[me.REGF],
                                       [1, 1, 2])]

        self.access[me.ITCN] = [v * u * f for v, u, f
                                in zip(self.unit_access[me.ITCN],
                                       total_units,
                                       self.fetch[self.BL.REGF])]

        self.access[me.GBUF] = [v * u * f * s for v, u, f, s
                                in zip(self.unit_access[me.GBUF],
                                       total_units,
                                       self.fetch[self.BL.REGF],
                                       self.stored_in_gbuf)]

        self.access[me.DRAM] = [v * u * f for v, u, f
                                in zip(self.unit_access[me.DRAM],
                                       total_units,
                                       self.fetch[self.BL.GBUF])]

    def _innermost_nontrivial_loop(self, bl_lvl):
        '''
        Get the innermost non-trivial loop at blocking level `bl_lvl`. Return
        None if all loops are trivial.

        The innermost non-trivial loop has a non-one blocking factor, and the
        smallest order value.
        '''
        # Order of the current level, indexed by LoopEnum.
        order = self.orders[bl_lvl]
        # If not all loops are trivial, the first element in the tuple will
        # pick them out (False < True). Then the second element returns the
        # smallest order value.
        # If all loops are trivial, the last tuple is the min one, which
        # returns None.
        return min((self.ti[bl_lvl] == 1, order[le.IFM], le.IFM),
                   (self.to[bl_lvl] == 1, order[le.OFM], le.OFM),
                   (self.tb[bl_lvl] == 1, order[le.BAT], le.BAT),
                   (False, float('inf'), None))[2]

