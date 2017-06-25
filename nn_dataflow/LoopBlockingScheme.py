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
import itertools

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
                 resource, part_occ, options):
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

        `part_occ` is the partitioning occupation.
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

        tps = [Util.prod(ts) for ts in self._bl_t(slice(None))]
        self.total_units = self._t_data_cnt(tps)

        self.lcnt = self.tip * self.top * self.tbp

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

        # Record unit stats.
        self.unit_ops = nested_loop_desc.unit_ops
        self.unit_time = nested_loop_desc.unit_time
        self.unit_access = nested_loop_desc.unit_access

        # Parallel partitioning.
        self.num_nodes = resource.dim_nodes.size()
        # Occupation.
        # Occupation only affects op counts and REGF accesses.
        self.part_occ = part_occ

        # Stats: lazy evaluation.
        self.finalized_stats = False
        self.ops = float('nan')
        self.time = float('nan')
        self.access = [[float('nan')] * de.NUM for _ in range(me.NUM)]

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

    def get_access(self):
        '''
        Get number of accesses of each data category to each hierarchy.

        Access is a two-dimensional list, first indexed by MemHierEnum, then
        indexed by DataCategoryEnum.
        '''
        if not self.is_valid():
            return None

        if not self.finalized_stats:
            self._calc_stats()

        return self.access

    def get_top_level_fetch(self):
        '''
        Get number of top-level-hierarchy fetches of each data category.
        '''
        if not self.is_valid():
            return None

        if not self.finalized_stats:
            self._calc_stats()

        return self.fetch[self.BL.GBUF]

    def get_cost(self, cost):
        '''
        Get the total cost of loop blocking.
        '''
        if not self.is_valid():
            return float('inf')

        if not self.finalized_stats:
            self._calc_stats()

        c = 0

        c += self.ops * cost.mac_op

        access_total = [sum(acc) for acc in self.access]
        c += sum(mc * ma for mc, ma in zip(cost.mem_hier, access_total))

        c += self.time * cost.unit_static * self.num_nodes

        return c

    def get_scheme_dict(self, cost):
        '''
        Get an OrderedDict of scheme summary.
        '''
        if not self.is_valid():
            return None

        if not self.finalized_stats:
            self._calc_stats()

        size = [[self.data_size(bl, dce) for dce in range(de.NUM)]
                for bl in range(self.BL.NUM)]

        return OrderedDict([('cost', self.get_cost(cost)),
                            ('ops', self.ops),
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

    def gen_index(self):
        '''
        Generate the indexes of ifmap, ofmap and batch sample, based on the
        loop blocking factors and the orders. Index will be 0 to total loop
        count, e.g., 0 to `loopcnt_ifm`.

        Return the indexes in the order of LoopEnum.
        '''

        # Index generators for all blocking levels.
        bl_idxgen_list = []
        # Counts of loop units for all blocking levels.
        bl_cnt_list = []

        assert self.BL.NUM == 2
        bl_gbuf = self.BL.GBUF
        bl_regf = self.BL.REGF

        # Between DRAM and GBUF.
        t_x = self._bl_t(bl_gbuf)
        order_x = self.orders[bl_gbuf]
        cnt_x = [Util.prod(ts) for ts in self._bl_t(slice(bl_gbuf + 1, None))]
        bl_idxgen_list.append(self._gen_index_single_level(t_x, order_x))
        bl_cnt_list.append(cnt_x)

        # Between GBUF and REGF.
        t_x = self._bl_t(bl_regf)
        order_x = self.orders[bl_regf]
        cnt_x = [Util.prod(ts) for ts in self._bl_t(slice(bl_regf + 1, None))]
        bl_idxgen_list.append(self._gen_index_single_level(t_x, order_x))
        bl_cnt_list.append(cnt_x)

        # Between REGF and ALU.
        t_x = self._bl_t(2)
        order_x = (0, 1, 2)
        cnt_x = (1,) * le.NUM
        bl_idxgen_list.append(self._gen_index_single_level(t_x, order_x))
        bl_cnt_list.append(cnt_x)

        # Generate.
        num = 0
        for bl_idx_list in itertools.product(*bl_idxgen_list):
            # Merge indexes of all levels.
            idx = (0,) * le.NUM

            # bl_idx_list is (i0, o0, b0), (i1, o1, b1), ...
            # bl_cnt_list is (tip0, top0, tbp0), (tip1, top1, tbp1), ...
            # idx should be (i0 * tip0 + i1 * tip1 + ..., ...)
            for bl_idx, bl_cnt in zip(bl_idx_list, bl_cnt_list):
                idx = tuple(i + bi * bc for i, bi, bc
                            in zip(idx, bl_idx, bl_cnt))

            num += 1
            yield idx

        assert num == self.lcnt

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
            bl_tps = [Util.prod(ts) for ts in self._bl_t(slice(blp1, None))]
            uc = self._t_data_cnt(bl_tps)
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
            innermost_nt_lp = self._innermost_nontrivial_loop(self._bl_t(bl),
                                                              self.orders[bl])

            cnt = self._t_data_cnt(self._bl_t(bl))

            fe = [0] * de.NUM

            for dce, lpe in zip([de.FIL, de.IFM, de.OFM],
                                [le.BAT, le.OFM, le.IFM]):
                if cnt[dce] == 1:
                    fe[dce] = self.fetch[bl-1][dce] if bl > 0 else 1
                else:
                    bl_start = bl + (innermost_nt_lp != lpe)
                    f = Util.prod(self._bl_t(slice(bl_start))[lpe])
                    fe[dce] = 2 * f - 1 if dce == de.OFM else f

            self.fetch.append(fe)

    def _calc_stats(self):
        '''
        Lazily calculate stats.
        '''

        self.ops = self.unit_ops * self.lcnt * self.num_nodes * self.part_occ
        self.time = self.unit_time * self.lcnt

        self.access[me.REGF] = [v * self.lcnt * t
                                * self.num_nodes * self.part_occ for v, t
                                in zip(self.unit_access[me.REGF],
                                       [1, 1, 2])]

        self.access[me.ITCN] = [v * u * f * self.num_nodes for v, u, f
                                in zip(self.unit_access[me.ITCN],
                                       self.total_units,
                                       self.fetch[self.BL.REGF])]

        self.access[me.GBUF] = [v * u * f * s * self.num_nodes for v, u, f, s
                                in zip(self.unit_access[me.GBUF],
                                       self.total_units,
                                       self.fetch[self.BL.REGF],
                                       self.stored_in_gbuf)]

        self.access[me.DRAM] = [v * u * f * self.num_nodes for v, u, f
                                in zip(self.unit_access[me.DRAM],
                                       self.total_units,
                                       self.fetch[self.BL.GBUF])]

        self.finalized_stats = True

    def _bl_t(self, blvl):
        '''
        Get the loop blocking factors of level `blvl`.
        '''
        bl_t = [0] * le.NUM
        bl_t[le.IFM] = self.ti[blvl]
        bl_t[le.OFM] = self.to[blvl]
        bl_t[le.BAT] = self.tb[blvl]
        return bl_t

    @staticmethod
    def _t_data_cnt(bl_t):
        '''
        Get the corresponding data unit counts given the loop blocking factors.
        '''
        cnt = [0] * de.NUM
        cnt[de.FIL] = bl_t[le.IFM] * bl_t[le.OFM]
        cnt[de.IFM] = bl_t[le.IFM] * bl_t[le.BAT]
        cnt[de.OFM] = bl_t[le.OFM] * bl_t[le.BAT]
        return cnt

    @staticmethod
    def _innermost_nontrivial_loop(bl_t, bl_ord):
        '''
        Get the innermost non-trivial loop at a level. Return None if all loops
        are trivial.

        The innermost non-trivial loop has a non-one blocking factor, and the
        smallest order value.

        `bl_t` are the loop blocking factors, indexed by LoopEnum. `bl_ord` is
        the loop order.
        '''
        # The key is a tuple.
        # The first element picks out the non-trivial loops (False < True). If
        # all loops are trivial, None will be the only left one.
        # The second element compares the order within the non-trivial loops.
        return min([None] + range(le.NUM),
                   key=lambda lpe: (bl_t[lpe] == 1, bl_ord[lpe]) \
                           if lpe is not None else (False, le.NUM))

    @staticmethod
    def _gen_index_single_level(t_x, order_x):
        '''
        Generate the indexes of a single loop blocking level.
        '''
        # The element in order is the position from inner to outer, we list the
        # generators from outer to inner.
        gens = [None] * le.NUM
        rev_order = [le.NUM - 1 - o for o in order_x]
        for lpe in range(le.NUM):
            gens[rev_order[lpe]] = xrange(t_x[lpe])

        for idx in itertools.product(*gens):
            # Index now is in the loop order from outer to inner. Reorder to be
            # in LoopEnum order.
            yield tuple(idx[rev_order[lpe]] for lpe in range(le.NUM))

