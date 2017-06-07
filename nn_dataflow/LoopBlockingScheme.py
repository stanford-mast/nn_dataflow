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
                 resource, bufshr, options):
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

        `bufshr` is a BufShrScheme instance, indicating the buffer sharing
        scheme.
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

        # Buffer sharing initialization.
        self._init_bufshr(bufshr, options)

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

        # Occupation.
        # Occupation only affects op counts and REGF accesses.
        self.part_occ = 1.  # set later.

        # Stats: lazy evaluation.
        self.finalized_stats = False
        self.ops = float('nan')
        self.time = float('nan')
        self.access = [[float('nan')] * de.NUM for _ in range(me.NUM)]
        # NoC access due to buffer sharing and access forwarding.
        self.noc_access = [0] * de.NUM

        # Buffer sharing.
        self._set_bufshr(resource, bufshr, options)
        if not self.is_valid():
            return

        # Access forwarding.
        self._set_accfwd(bufshr, options)

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
            size = Util.idivc(size, self.bufshr_subgrp_size[dce])

        return size

    def set_partition_occupation(self, part_occ):
        ''' Set and scale by the given partitioning occupation. '''
        if not self.is_valid():
            return
        assert not self.finalized_stats
        self.part_occ = part_occ

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

    def get_noc_access(self):
        '''
        Get the NoC accesses of each data category.
        '''
        if not self.is_valid():
            return None

        if not self.finalized_stats:
            self._calc_stats()

        return self.noc_access

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

        c += sum(self.noc_access) * cost.noc_hop

        c += self.time * cost.unit_static

        return c

    def get_scheme_dict(self):
        '''
        Get an OrderedDict of scheme summary.
        '''
        if not self.is_valid():
            return None

        if not self.finalized_stats:
            self._calc_stats()

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
                            ('orders', self.orders),
                            ('accfwd_reduction', self.accfwd_reduction),
                            ('bufshr_grp_size', self.bufshr_grp_size),
                            ('bufshr_subgrp_size', self.bufshr_subgrp_size),
                            ('bufshr_rot_fetch', self.bufshr_rot_fetch),
                            ('bufshr_rot_round_cnt', self.bufshr_rot_round_cnt),
                            ('bufshr_rot_unit_cnt', self.bufshr_rot_unit_cnt),
                            ('bufshr_wide_fetch', self.bufshr_wide_fetch),
                            ('bufshr_wide_fetch_width', self.bufshr_wide_fetch_width),
                           ])

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

        self.ops = self.unit_ops * self.lcnt * self.part_occ
        self.time = self.unit_time * self.lcnt

        self.access[me.REGF] = [v * self.lcnt * t * self.part_occ for v, t
                                in zip(self.unit_access[me.REGF],
                                       [1, 1, 2])]

        self.access[me.ITCN] = [v * u * f for v, u, f
                                in zip(self.unit_access[me.ITCN],
                                       self.total_units,
                                       self.fetch[self.BL.REGF])]

        self.access[me.GBUF] = [v * u * f * s for v, u, f, s
                                in zip(self.unit_access[me.GBUF],
                                       self.total_units,
                                       self.fetch[self.BL.REGF],
                                       self.stored_in_gbuf)]

        self.access[me.DRAM] = [v * u * f / r for v, u, f, r
                                in zip(self.unit_access[me.DRAM],
                                       self.total_units,
                                       self.fetch[self.BL.GBUF],
                                       self.accfwd_reduction)]

        # NoC access.
        bufshr_rot_access = self._calc_bufshr_rotation_access(
            self.bufshr_rot_fetch)
        bufshr_wf_access = self._calc_bufshr_widefetch_access(
            self.bufshr_wide_fetch)
        self.noc_access = [a1 + a2 for a1, a2
                           in zip(bufshr_rot_access, bufshr_wf_access)]

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

    def verify_fetch(self):
        '''
        Verify the data fetch times, by actually simulating and generating the
        loops.
        '''

        if not self.is_valid():
            return

        # Counts of loop units for all blocking levels.
        bl_cnt_list = []
        for bl in range(self.BL.NUM):
            cnt = [0] * de.NUM
            cnt_ifm = Util.prod(self.ti[bl+1:])
            cnt_ofm = Util.prod(self.to[bl+1:])
            cnt_bat = Util.prod(self.tb[bl+1:])
            cnt[de.FIL] = (cnt_ifm, cnt_ofm)
            cnt[de.IFM] = (cnt_ifm, cnt_bat)
            cnt[de.OFM] = (cnt_ofm, cnt_bat)
            bl_cnt_list.append(cnt)

        # Buffered data ranges.
        # The range of each data category is a pair of ranges. Each range is the
        # index range of one dimension. E.g., for IFM, the first is for i's,
        # and the second is for b's. The range is represented by a pair of
        # start and end.
        gbuf_data = [((0, 0), (0, 0)) for _ in range(de.NUM)]
        regf_data = [((0, 0), (0, 0)) for _ in range(de.NUM)]

        # Accesses to upper level.
        gbuf_access = [0] * de.NUM
        regf_access = [0] * de.NUM

        def _replace(data, access, dce, idx_pr, cnt_pr, buf_cnt_pr,
                     bypass=False):
            '''
            Replace the data `dce` buffered in `data` and increment `access`.

            Return the count for all dimensions of the accessed data.

            `idx_pr` and `cnt_pr` are the index and count for all dimensions of
            the accessed data. `buf_cnt_pr` is the count for all dimensions of
            the buffered data.
            '''
            hit = all(rngs[0] <= idx < rngs[1]
                      for idx, rngs in zip(idx_pr, data[dce]))
            if not hit:
                if bypass:
                    # Bypass.
                    access[dce] += Util.prod(cnt_pr)
                    return cnt_pr

                # Miss.
                idxb_pr = [idx // cnt * cnt
                           for idx, cnt in zip(idx_pr, buf_cnt_pr)]
                data[dce] = tuple((ib, ib + cnt)
                                  for ib, cnt in zip(idxb_pr, buf_cnt_pr))
                access[dce] += Util.prod(buf_cnt_pr)
                return buf_cnt_pr

            # Hit.
            return (0, 0)

        for iidx, oidx, bidx in self.gen_index():

            for dce, idx_pr in zip([de.FIL, de.IFM, de.OFM],
                                   [(iidx, oidx), (iidx, bidx), (oidx, bidx)]):
                cnt = (1, 1)

                # REGF.
                cnt = _replace(regf_data, regf_access, dce, idx_pr,
                               cnt, bl_cnt_list[self.BL.REGF][dce])
                if not any(cnt):
                    continue

                # GBUF.
                cnt = _replace(gbuf_data, gbuf_access, dce, idx_pr,
                               cnt, bl_cnt_list[self.BL.GBUF][dce],
                               not self.stored_in_gbuf[dce])

        if not all(a % u == 0 for a, u in zip(gbuf_access, self.total_units)):
            raise RuntimeError('LoopBlockingScheme: fetch verification failed. '
                               'GBUF access is not multiple of total units. '
                               'access {}, units {}.'
                               .format(gbuf_access, self.total_units))
        if not all(a % u == 0 for a, u in zip(regf_access, self.total_units)):
            raise RuntimeError('LoopBlockingScheme: fetch verification failed. '
                               'REGF access is not multiple of total units. '
                               'access {}, units {}.'
                               .format(regf_access, self.total_units))

        # Fetch times to upper level.
        gbuf_fetch = [a // u for a, u in zip(gbuf_access, self.total_units)]
        regf_fetch = [a // u for a, u in zip(regf_access, self.total_units)]

        # Output is read/write.
        gbuf_fetch[de.OFM] = 2 * gbuf_fetch[de.OFM] - 1
        regf_fetch[de.OFM] = 2 * regf_fetch[de.OFM] - 1

        # Verify.
        if not all(f1 == f2 for f1, f2
                   in zip(gbuf_fetch, self.fetch[self.BL.GBUF])):
            raise RuntimeError('LoopBlockingScheme: fetch verification failed. '
                               'GBUF fetch mismatch. model {}, sim {}, '
                               'blocking {}, orders {}.'
                               .format(self.fetch[self.BL.GBUF], gbuf_fetch,
                                       (self.ti, self.to, self.tb),
                                       self.orders))
        if not all(f1 == f2 for f1, f2
                   in zip(regf_fetch, self.fetch[self.BL.REGF])):
            raise RuntimeError('LoopBlockingScheme: fetch verification failed. '
                               'REGF fetch mismatch. model {}, sim {}, '
                               'blocking {}, orders {}.'
                               .format(self.fetch[self.BL.REGF], regf_fetch,
                                       (self.ti, self.to, self.tb),
                                       self.orders))

    def _set_accfwd(self, bufshr, options):
        '''
        Set access forwarding (AF).
        '''
        assert self.is_valid() and not self.finalized_stats

        # DRAM access reduction due to AF. This is the average reduction. Each
        # node does not need to fetch exactly 1/N data.
        self.accfwd_reduction = [1] * de.NUM

        if not options.hw_access_forwarding and not options.hw_gbuf_sharing:
            return

        # If n nodes share the data, each node fetches 1/n of the data.
        for dce in range(de.NUM):
            self.accfwd_reduction[dce] = bufshr.size(dce)

    def _init_bufshr(self, bufshr, options):
        '''
        Initialize buffer sharing (BS).

        Must be called before any buffered data size check.
        '''
        assert not hasattr(self, "unit_cnt")

        # Total BS nodes
        self.bufshr_grp_size = tuple(bufshr.size(dce) if options.hw_gbuf_sharing
                                     else 1 for dce in range(de.NUM))
        # BS subgroup sizes.
        # The initial values are conservative, i.e., assuming the maximum
        # shared capacity across nodes.
        # They can be decreased later, but never increased.
        self.bufshr_subgrp_size = self.bufshr_grp_size

        # NoC fetch due to rotation.
        # The fetch times means the number of hops along which each data
        # (considered all replica) traverals over the entire nested loops.
        # The total number of hops of all data over all nodes will be this
        # value multiplying the size of unique data (without replica).
        self.bufshr_rot_fetch = [0.] * de.NUM
        # Rotation round counts.
        self.bufshr_rot_round_cnt = [0] * de.NUM
        # Rotation unit counts.
        self.bufshr_rot_unit_cnt = [float('nan')] * de.NUM

        # NoC fetch due to wide fetch. Meaning similar to `bufshr_rot_fetch`.
        self.bufshr_wide_fetch = [0.] * de.NUM
        # Wide fetch widths.
        self.bufshr_wide_fetch_width = [0.] * de.NUM

    def _set_bufshr(self, resource, bufshr, options):
        '''
        Set buffer sharing (BS).

        The GBUF level loops, i.e., ti/to/tb[1], decide the order and ranges of
        the access to data buffered in GBUF, which could spread across multiple
        nodes. Depending on the loop structure, at most one data category,
        whose two related loops are not adjacent and split by the other loop,
        has a non-perfect-sequential access pattern, as the inner dimension
        will be accessed multiple times (due to the middle unrelated loop)
        before switching to the next outer dimension. We call it non-seq-acc
        data category. If there are < 3 non-trivial loops, there is no non-seq
        data category. E.g., OFM is non-seq-acc with the following loop
        structure:

        for o
          for i
            for b

        - Rotation round.

        The blocking factors and loop order decide the number of rotation
        rounds. For seq-acc data categories, the rotation rounds equal to the
        fetch times to GBUF. E.g., with above loops, IFM (i, b) rotates `to`
        rounds, and FIL (i, o) rotates once. For non-seq-acc data category, we
        only rotate after all the multiple accesses to the inner dimension are
        done. So its rotation rounds needs to be reduced by this fetch times.
        E.g., OFM (o, b) rotates only once.

        - Rotation unit.

        Rotation unit for each data category is defined as the shifting size
        for each rotation step. For seq-acc data categories, the rotation unit
        is 1 REGF unit. For non-seq-acc data category, the rotation unit is 1
        REGF unit * inner dimension size.

        - Wide fetch.

        Rotation unit size does not affect the NoC access of rotation rounds,
        but there may be remote accesses without rotation, called wide fetch,
        if the rotation unit does not fit in a single node GBUF.

        - Rotation round optimization.

        Only the data with the inner two loops are rotated multiple times per
        DRAM access (GBUF filling) and can be optimized. We do not touch the
        innermost loop, to keep GBUF fetch times unchanged. Therefore we only
        try to split the middle loop into two: across-node and within-node, and
        bring the across-node loop outside.

        With above loop structure example, it will be

        for i-across-node
          for o
            for i-within-node
              for b

        This optimization reduces IFM (i, b) rotation rounds from `to` to 1,
        and increases OFM (o, b) rotation rounds from 1 to `i-across-node`,
        i.e., subgroup size of IFM; it does not change FIL (i, o) rotation
        rounds.

        This optimization does not change the wide fetch times. FIL (i, o) is
        still seq-acc once but with a different order; OFM (o, b) is still
        non-seq-acc; IFM (i, b) becomes non-seq-acc, but if we ensure the inner
        i loop is within a node, no wide fetch is needed.
        '''
        assert self.is_valid() and not self.finalized_stats

        if not options.hw_gbuf_sharing:
            return

        bl = self.BL.GBUF
        blp1 = bl + 1

        # If bypass gbuf, set subgroup size to 1.
        self.bufshr_subgrp_size = tuple(sgs if self.stored_in_gbuf[dce]
                                        and self.unit_size[bl][dce] > 0 else 1
                                        for dce, sgs
                                        in enumerate(self.bufshr_subgrp_size))

        # The blocking factors and loop order that are related to BS.
        t_bs = self._bl_t(blp1)
        ord_bs = self.orders[blp1]

        # Non-trivial loops.
        nt_loops_bs = set(lpe for lpe in range(le.NUM) if t_bs[lpe] > 1)
        inlp_bs = self._innermost_nontrivial_loop(t_bs, ord_bs)

        def _min_subgrp_size(*dce_list):
            '''
            Get the minimum BS subgroup size. Minimize in the order of the
            given `dce_list`.
            '''
            cur_sgs = list(self.bufshr_subgrp_size)
            min_sgs = list(self.bufshr_subgrp_size)
            free_cap = resource.size_gbuf - self.data_size(bl)
            for dce in dce_list:
                dce_size = self.data_size(bl, dce)
                assert dce_size > 0
                dce_tot_size = dce_size * cur_sgs[dce]
                # dce_tot_size / sgs - dce_size <= free_cap
                min_sgs[dce] = Util.idivc(dce_tot_size, free_cap + dce_size)
                free_cap -= dce_tot_size / min_sgs[dce] - dce_size
                assert free_cap >= 0
            return tuple(min_sgs)

        def _rotation(ord_loops):
            '''
            Get the rotation information.

            Return the number of rotation rounds and the number of rotation
            units for each data category.
            '''
            rotrnd_cnt = list(self.fetch[blp1])
            rotunit_cnt = self._t_data_cnt(t_bs)

            # The non-seq-acc data category with nonadjacent loops.
            nseq_dce = None
            if len(ord_loops) == 3:
                nseq_dce = (de.FIL if ord_loops[1] == le.BAT
                            else (de.IFM if ord_loops[1] == le.OFM
                                  else de.OFM))
                # Update rotation unit to be the whole inner dim.
                rotunit_cnt[nseq_dce] //= t_bs[ord_loops[2]]
                # Reduce rotation rounds by the fetch times to the inner dim.
                rotrnd_cnt[nseq_dce] //= t_bs[ord_loops[1]]

            return rotrnd_cnt, rotunit_cnt, nseq_dce

        def _sweep_rotation():
            '''
            Generate all potential rotation schemes.

            Yield the resulting NoC rotation fetch times, and the rotation
            scheme.
            '''
            for non_inlp in itertools.permutations([lpe for lpe in nt_loops_bs
                                                    if lpe != inlp_bs]):
                # Ordered loops, from outermost to innermost.
                ord_loops = non_inlp + (inlp_bs,)

                rotrnd_cnt, rotunit_cnt, nseq_dce = _rotation(ord_loops)

                # Reduce subgroup size if data can fit in fewer nodes. Need to
                # decide an order about which data first shrink.
                dce_list = [dce for dce in range(de.NUM)
                            if self.bufshr_subgrp_size[dce] > 1]
                for dce_order in itertools.permutations(dce_list):
                    subgrp_size = _min_subgrp_size(*dce_order)

                    # Wide fetch.
                    wf_width = [1. * s / c for s, c
                                in zip(subgrp_size, rotunit_cnt)]
                    fetch_per_bufacc = [
                        bufshr.nhops_wide_fetch_once(dce, subgrp_size[dce],
                                                     wf_width[dce])
                        for dce in range(de.NUM)]
                    # Wide fetch counts, equal to GBUF access (REGF filling).
                    bufacc_cnt = self.fetch[blp1]
                    # The last wide fetch can be combined with the rotation.
                    wide_fetch = [nh * (r - f) for nh, r, f
                                  in zip(fetch_per_bufacc, bufacc_cnt,
                                         rotrnd_cnt)]
                    assert all(wf >= 0 - 1e-4 for wf in wide_fetch)

                    # Rotation.
                    fetch_per_rot = [
                        bufshr.nhops_rotate_all(dce, subgrp_size[dce])
                        for dce in range(de.NUM)]
                    # The first rotation can be combined with the initial
                    # broadcast.
                    # After fetching from DRAM, data will be broadcast to all
                    # nodes regardless of who stores it (see Partition). So
                    # initially each node receives all data. This saves one
                    # rotation. Afterwards each node only stores partial data
                    # and relies on rotation to see all the data.
                    rot_fetch = [nh * (r - f) for nh, r, f
                                 in zip(fetch_per_rot, rotrnd_cnt,
                                        self.fetch[bl])]
                    assert all(rf >= 0 - 1e-4 for rf in rot_fetch)

                    yield rot_fetch, wide_fetch, subgrp_size, \
                            rotrnd_cnt, rotunit_cnt, wf_width

                    if len(ord_loops) == 3:
                        # Optimize rotation rounds.
                        rotrnd_cnt_opt = list(rotrnd_cnt)
                        # The target data category is the one with the inner
                        # two loops.
                        dce_to_opt = (de.FIL if ord_loops[0] == le.BAT
                                      else (de.IFM if ord_loops[0] == le.OFM
                                            else de.OFM))
                        rotrnd_cnt_opt[dce_to_opt] //= t_bs[ord_loops[0]]
                        # And the non-seq-acc data rotation rounds increase.
                        rotrnd_cnt_opt[nseq_dce] *= subgrp_size[dce_to_opt]

                        rot_fetch_opt = [nh * (r - f) for nh, r, f
                                         in zip(fetch_per_rot, rotrnd_cnt_opt,
                                                self.fetch[bl])]
                        assert all(rf >= 0 - 1e-4 for rf in rot_fetch_opt)

                        yield rot_fetch_opt, wide_fetch, subgrp_size, \
                                rotrnd_cnt_opt, rotunit_cnt, wf_width

        try:
            def _key_func(tuple_):
                return sum(self._calc_bufshr_rotation_access(tuple_[0])) \
                        + sum(self._calc_bufshr_widefetch_access(tuple_[1]))
            rot_fetch, wide_fetch, subgrp_size, \
                    rotrnd_cnt, rotunit_cnt, wf_width \
                    = min(_sweep_rotation(), key=_key_func)
        except ValueError:
            self.valid = False
            return

        self.bufshr_rot_fetch = rot_fetch
        self.bufshr_rot_round_cnt = rotrnd_cnt
        self.bufshr_rot_unit_cnt = rotunit_cnt
        self.bufshr_wide_fetch = wide_fetch
        self.bufshr_widefetch_width = wf_width
        self.bufshr_subgrp_size = subgrp_size

    def _calc_bufshr_rotation_access(self, bufshr_rot_fetch):
        ''' Calculate the BS rotation NoC accesses. '''
        # Per-node access needs to divide by group size.
        # See the definition of bufshr_rot_fetch.
        return [v * u * f / r for v, u, f, r
                in zip(self.unit_access[me.GBUF], self.total_units,
                       bufshr_rot_fetch, self.bufshr_grp_size)]

    def _calc_bufshr_widefetch_access(self, bufshr_wide_fetch):
        ''' Calculate the BS wide fetch NoC accesses. '''
        # Per-node access needs to divide by group size.
        return [v * u * f / r for v, u, f, r
                in zip(self.unit_access[me.GBUF], self.total_units,
                       bufshr_wide_fetch, self.bufshr_grp_size)]

