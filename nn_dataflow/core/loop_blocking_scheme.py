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

import itertools
import math

from . import data_category_enum as de
from . import loop_enum as le
from . import mem_hier_enum as me
from .node_region import NodeRegion
from .. import util

class LoopBlockingScheme():
    '''
    Loop blocking scheme.

    Consider the loops of ifmap, ofmap, and batching.
    '''
    # pylint: disable=too-many-instance-attributes

    class BL():  # pylint: disable=too-few-public-methods
        '''
        Blocking-level enum. Only used locally.
        '''
        GBUF = 0
        REGF = 1
        NUM = 2

    def __init__(self, nested_loop_desc, bl_ts, bl_ords, resource, bufshr,
                 options):
        '''
        Given blocking factors `bl_ts` and the loop orders `bl_ords`, construct
        the loop blocking scheme.

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

        `bl_ts` are the blocking factors of all levels, indexed by BL, but with
        length of `BL.NUM + 1`, where the last entry corresponds to the
        computation within one PE. Each entry is a tuple indexed by LoopEnum
        and gives the loop blocking factors at this level.

        `bl_ords` indicate the loop orders of all levels, indexed by BL. Each
        entry is a permutation tuple indexed by LoopEnum and gives the
        positions of the loops at this level. Smaller number means inner loop.

        `bufshr` is a BufShrScheme instance, indicating the buffer sharing
        scheme.
        '''

        # pylint: disable=invalid-name
        BL = self.BL

        # Loop structure.
        self.nld = nested_loop_desc
        # Cache values.
        self.total_access_gbuf = [self.nld.total_access_at_of(me.GBUF, dce)
                                  for dce in range(de.NUM)]

        # Check lengths and values.
        assert len(bl_ts) == BL.NUM + 1, \
                'LoopBlockingScheme: bl_ts has invalid length.'
        assert all(len(bl_t) == le.NUM for bl_t in bl_ts), \
                'LoopBlockingScheme: bl_ts elements have invalid length.'
        assert len(bl_ords) == BL.NUM, \
                'LoopBlockingScheme: bl_ords has invalid length.'
        assert all(tuple(sorted(bl_ord)) == tuple(range(le.NUM)) \
                   for bl_ord in bl_ords), \
                'LoopBlockingScheme: bl_ords elements are invalid.'

        self.bl_ts = [tuple(bl_t) for bl_t in bl_ts]
        self.bl_ords = [tuple(bl_ord) for bl_ord in bl_ords]

        # Check blocking.
        bl_tp = self._bl_tp(slice(None))
        for lpe in range(le.NUM):
            assert bl_tp[lpe] >= self.nld.loopcnt[lpe], \
                    'LoopBlockingScheme: invalid blocking LP {}: {} for {}.' \
                    .format(lpe, self.bl_ts, self.nld.loopcnt)

        self.lcnt = util.prod(bl_tp)

        # Need to define time for invalid scheme.
        self.time = float('inf')

        # Buffer sharing initialization.
        self._init_bufshr(bufshr, options)

        # Buffer data size for one unit.
        self.unit_size = [tuple() for _ in range(BL.NUM)]
        self.unit_size[BL.GBUF] = self.nld.usize_gbuf
        self.unit_size[BL.REGF] = self.nld.usize_regf

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
        self.valid = True

        # Data fetch calculation.
        self._set_fetch()

        # Check resource data src/dst region.
        self.src_is_dram = (resource.src_data_region.type == NodeRegion.DRAM)
        self.dst_is_dram = (resource.dst_data_region.type == NodeRegion.DRAM)

        self.filter_pinned = False

        # If data regions are not DRAM, can only access once, no spilling.
        if not self.src_is_dram:
            if self.fetch[BL.GBUF][de.IFM] > 1:
                self.valid = False
                return
            if resource.src_data_region == resource.proc_region:
                # Force to store in gbuf.
                self.stored_in_gbuf[de.IFM] = True
        if not self.dst_is_dram:
            if self.fetch[BL.GBUF][de.OFM] > 1:
                self.valid = False
                return
            if resource.dst_data_region == resource.proc_region:
                # Force to store in gbuf.
                self.stored_in_gbuf[de.OFM] = True

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

        # Array bus.
        self.array_bus_width = resource.array_bus_width
        # DRAM bandwidth.
        self.dram_bandwidth = resource.dram_bandwidth
        # Parallel partitioning.
        self.num_nodes = resource.proc_region.dim.size()

        # Stats: lazy evaluation.
        self.finalized_stats = False
        self.ops = float('nan')
        self.time = float('nan')
        self.proc_time = float('nan')
        self.bus_time = float('nan')
        self.dram_time = float('nan')
        self.access = [[float('nan')] * de.NUM for _ in range(me.NUM)]

        # NoC access due to buffer sharing.
        self.noc_access = [0.] * de.NUM
        self.bufshr_rotation_access = [0.] * de.NUM
        self.bufshr_wide_fetch_access = [0.] * de.NUM

        # Buffer sharing.
        self._set_bufshr(resource, bufshr, options)

        # Access forwarding.
        self._set_accfwd(bufshr, options)

        # Remote gbuf access.
        self.remote_gbuf_access = [0.] * de.NUM

        # Check resource for filter pinning.
        if resource.no_time_mux:
            if all(self.bl_ts[0][lpe] == 1 for lpe
                   in self.nld.data_loops[de.FIL].loops()):
                self.filter_pinned = True
                self.fetch[0][de.FIL] = 0

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
            size = util.idivc(size, self.bufshr_subgrp_size[dce])

        return size

    def get_access(self):
        '''
        Get number of accesses of each data category to each hierarchy.

        Access is a two-dimensional list, first indexed by MemHierEnum, then
        indexed by DataCategoryEnum.
        '''
        if not self.is_valid():
            return [[float('inf')] * de.NUM for _ in range(me.NUM)]

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

    def get_access_cost(self, cost):
        '''
        Get the data access cost of loop blocking.
        '''
        if not self.is_valid():
            return float('inf')

        if not self.finalized_stats:
            self._calc_stats()

        acc_cost = sum(c * sum(a) for c, a in zip(cost.mem_hier, self.access))
        acc_cost += cost.mem_hier_at(me.GBUF) * sum(self.remote_gbuf_access)

        return acc_cost

    def gen_index(self):
        '''
        Generate the indexes of ifmap, ofmap and batch sample, based on the
        loop blocking factors and the orders. Index will be 0 to total loop
        count, e.g., 0 to `loopcnt[le.IFM]`.

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
        t_x = self.bl_ts[bl_gbuf]
        order_x = self.bl_ords[bl_gbuf]
        cnt_x = self._bl_tp(slice(bl_gbuf + 1, None))
        bl_idxgen_list.append(self._gen_index_single_level(t_x, order_x))
        bl_cnt_list.append(cnt_x)

        # Buffer sharing.
        t_x = self.bufshr_bs_t
        order_x = self.bufshr_bs_ord
        cnt_x = [x // b for x, b
                 in zip(self._bl_tp(slice(bl_gbuf + 1, None)),
                        self.bufshr_bs_t)]
        bl_idxgen_list.append(self._gen_index_single_level(t_x, order_x))
        bl_cnt_list.append(cnt_x)

        # Between GBUF and REGF.
        t_x = [x // b for x, b
               in zip(self.bl_ts[bl_regf], self.bufshr_bs_t)]
        order_x = self.bl_ords[bl_regf]
        cnt_x = self._bl_tp(slice(bl_regf + 1, None))
        bl_idxgen_list.append(self._gen_index_single_level(t_x, order_x))
        bl_cnt_list.append(cnt_x)

        # Between REGF and ALU.
        t_x = self.bl_ts[2]
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

    @classmethod
    def ordered_loops(cls, bl_t, bl_ord, lpe_only=False, reverse=False):
        '''
        Get the ordered loops from outermost to innermost according to the loop
        blocking factors `bl_t` and the loop order `bl_ord`, both indexed by
        LoopEnum. Trivial loops are ignored.

        If `reverse` is True, ordering the loops from innermost to outermost.

        Return a list of pairs of LoopEnum and blocking factor. If `lpe_only`
        is True, return a list of LoopEnum only.
        '''
        ord_lpes = list(sorted([lpe for lpe in range(le.NUM) if bl_t[lpe] > 1],
                               key=(lambda lpe: bl_ord[lpe]),
                               reverse=not reverse))
        if not lpe_only:
            return [(lpe, bl_t[lpe]) for lpe in ord_lpes]
        return ord_lpes

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
            uc = self._t_data_cnt(self._bl_tp(slice(bl + 1, None)))
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

            # A data category has related (dimension) loops and unrelated
            # loops. The data reuse only includes the current level blocking
            # factors of the unrelated loops if they are inner than all the
            # dimension loops, i.e., inner than the innermost non-trivial
            # dimension loop of the current level. If all the dimension loops
            # are trivial, i.e., have blocking factor 1, the current level does
            # not change the data, and the fetch times at this level is the
            # same as the outer level.

            fe = [0] * de.NUM

            bl_t = self.bl_ts[bl]
            bl_ord = self.bl_ords[bl]

            for dce in range(de.NUM):

                inntdim_lp = self._innt_dim_loop(dce, bl_t, bl_ord)

                if inntdim_lp is None:
                    fe[dce] = self.fetch[bl-1][dce] if bl > 0 else 1
                    continue

                f = 1
                for lpe in self.nld.data_loops[dce].drop(range(le.NUM)):
                    # Include the unrelated loop blocking factors outside of
                    # the innermost non-trivial dimension loop.
                    bl_start = bl + (bl_ord[lpe] > bl_ord[inntdim_lp])
                    f *= self._bl_tp(slice(bl_start))[lpe]

                fe[dce] = 2 * f - 1 if dce == de.OFM else f

            self.fetch.append(fe)

    def _calc_stats(self):
        '''
        Lazily calculate stats.
        '''

        self.ops = self.nld.unit_ops * self.lcnt * self.num_nodes
        self.proc_time = self.nld.unit_time * self.lcnt

        self.access[me.REGF] = [v * self.lcnt * t * self.num_nodes
                                for v, t in zip(self.nld.unit_access[me.REGF],
                                                [1, 1, 2])]

        self.access[me.ITCN] = [self.nld.total_access_at_of(me.ITCN, dce)
                                * self.fetch[self.BL.REGF][dce]
                                * self.num_nodes
                                for dce in range(de.NUM)]

        self.access[me.GBUF] = [self.nld.total_access_at_of(me.GBUF, dce)
                                * self.fetch[self.BL.REGF][dce]
                                * self.stored_in_gbuf[dce]
                                * self.num_nodes
                                for dce in range(de.NUM)]

        self.access[me.DRAM] = [(self.nld.total_access_at_of(me.DRAM, dce)
                                 if self.stored_in_gbuf[dce]
                                 else self.nld.total_access_at_of(me.GBUF, dce))
                                * self.fetch[self.BL.GBUF][dce]
                                * self.num_nodes
                                / self.accfwd_reduction[dce]
                                for dce in range(de.NUM)]

        # NoC access.
        self.bufshr_rotation_access = self._calc_bufshr_rotation_access(
            self.bufshr_rot_fetch)
        self.bufshr_wide_fetch_access = self._calc_bufshr_widefetch_access(
            self.bufshr_wide_fetch)
        self.noc_access = [a1 + a2 for a1, a2
                           in zip(self.bufshr_rotation_access,
                                  self.bufshr_wide_fetch_access)]

        if not self.src_is_dram:
            self.remote_gbuf_access[de.IFM] += self.access[me.DRAM][de.IFM]
            self.access[me.DRAM][de.IFM] = 0
        if not self.dst_is_dram:
            self.remote_gbuf_access[de.OFM] += self.access[me.DRAM][de.OFM]
            self.access[me.DRAM][de.OFM] = 0
        if self.filter_pinned:
            assert self.access[me.DRAM][de.FIL] == 0

        # DRAM access time.
        self.dram_time = int(math.ceil(sum(self.access[me.DRAM])
                                       / self.dram_bandwidth))

        # Array multicast uses separate bus for each data category.
        # Each data from GBUF takes one cycle to multicast to PEs.
        self.bus_time = util.idivc(int(math.ceil(1. * max(self.access[me.GBUF])
                                                 / self.num_nodes)),
                                   self.array_bus_width)

        # Optimistically assume processing, multicast, and DRAM access are well
        # overlapped, and ignore ramp-up/down.
        self.time = max(self.proc_time, self.bus_time, self.dram_time)

        self.finalized_stats = True

    def _bl_tp(self, bl_lvls):
        '''
        Get the products of the loop blocking factors for the given levels
        `bl_lvls`.
        '''
        assert isinstance(bl_lvls, slice)
        return [util.prod(ts[bl_lvls]) for ts in zip(*self.bl_ts)]

    def _t_data_cnt(self, bl_t):
        '''
        Get the corresponding data unit counts given the loop blocking factors.

        `bl_t` are the loop blocking factors, indexed by LoopEnum.
        '''
        return [util.prod(self.nld.data_loops[dce].take(bl_t))
                for dce in range(de.NUM)]

    def _innt_dim_loop(self, dce, bl_t, bl_ord):
        '''
        Get the innermost non-trivial loop which is a dimension loop of the
        given data category. Return None if all dimension loops are trivial.

        The innermost non-trivial dimension loop is one of the data dimension
        loops of data category `dce`, has a non-one blocking factor, and has
        the smallest order value.

        `bl_t` are the loop blocking factors, indexed by LoopEnum. `bl_ord` is
        the loop order.
        '''
        # The key is a tuple.
        # The first element picks out the non-trivial loops (False < True). If
        # all loops are trivial, None will be the only left one.
        # The second element compares the order within the non-trivial loops.
        return min((None,) + self.nld.data_loops[dce].loops(),
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
            gens[rev_order[lpe]] = range(t_x[lpe])

        for idx in itertools.product(*gens):
            # Index now is in the loop order from outer to inner. Reorder to be
            # in LoopEnum order.
            yield tuple(idx[rev_order[lpe]] for lpe in range(le.NUM))

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

        # Additional BS level between DRAM and GBUF, split out from GBUF level.
        self.bufshr_bs_t = (1,) * le.NUM
        self.bufshr_bs_ord = tuple(range(le.NUM))

        # NoC fetch due to rotation.
        # The fetch times means the number of hops along which each data
        # (considered all replica) traverals over the entire nested loops.
        # The total number of hops of all data over all nodes will be this
        # value multiplying the size of unique data (without replica).
        self.bufshr_rot_fetch = [0.] * de.NUM
        # Rotation round counts.
        self.bufshr_rot_round_cnt = [0] * de.NUM
        # Rotation unit counts.
        self.bufshr_rot_unit_cnt = [1] * de.NUM

        # NoC fetch due to wide fetch. Meaning similar to `bufshr_rot_fetch`.
        self.bufshr_wide_fetch = [0.] * de.NUM
        # Wide fetch widths.
        self.bufshr_wide_fetch_width = [0.] * de.NUM

    def _set_bufshr(self, resource, bufshr, options):
        '''
        Set buffer sharing (BS).

        The GBUF level loops, i.e., ti/to/tb[1], decide the order and ranges of
        the access to data buffered in GBUF, which could spread across multiple
        nodes.

        - Seq-acc and non-seq-acc data category.

        Depending on the loop structure, some data categories, whose related
        loops are not adjacent and split by the other unrelated loops, has a
        non-perfect-sequential access pattern, as the inner dimensions will be
        accessed multiple times (due to the middle unrelated loops) before
        switching to the next outer dimension. We call it non-seq-acc data
        category.

        E.g., with CONV layer, OFM is non-seq-acc with the following loop
        structure:

        for o
          for i
            for b

        If there are < 3 non-trivial loops, there is no non-seq data category.

        - Rotation unit.

        Rotation unit for each data category is defined as the shifting size
        for each rotation step. For seq-acc data categories, the rotation unit
        is single REGF unit. For non-seq-acc data category, the rotation unit
        is the product of all inner dimension sizes that are not adjacent to
        the outermost dimension, i.e., we only rotate after all the multiple
        accesses to the inner dimensions are done.

        - Rotation round.

        Given the definition of rotation unit above, the number of rotation
        rounds is the product of all unrelated loop blocking factors above the
        outermost dimension loop of this data category.

        E.g., with above loops, IFM (i, b) rotates `to` rounds, FIL (i, o)
        rotates once, and OFM (o, b) rotates only once.

        - Wide fetch.

        Rotation unit size does not affect the NoC access of rotation rounds,
        but there may be remote accesses without rotation, called wide fetch,
        if the rotation unit does not fit in a single node GBUF.

        - BS schemes.

        When exploring the BS schemes, we keep the total accesses to DRAM,
        GBUF, and REGF unchanged, i.e., previously calculated fetch times are
        still valid. This is guaranteed by fixing some innermost loops in the
        GBUF level.

        The other un-fixed loops (we call them flexible loops) can be reordered
        or further blocked into an additional BS level between GBUF and DRAM
        levels. This additional level can help reduce NoC accesses by splitting
        the data accesses into across-node and within-node, and use up the data
        within a node before switching to the next node.

        E.g., the above loop structure can become:

        for i-across-node
          for o
            for i-within-node
              for b

        This optimization reduces IFM (i, b) rotation rounds from `to` to 1,
        and increases OFM (o, b) rotation rounds from 1 to `i-across-node`,
        i.e., subgroup size of IFM; it does not change FIL (i, o) rotation
        rounds.
        '''
        assert self.is_valid() and not self.finalized_stats

        if not options.hw_gbuf_sharing:
            assert all(gs == 1 for gs in self.bufshr_grp_size)
            return

        bl = self.BL.GBUF
        blp1 = bl + 1

        # If bypass GBUF, set subgroup size to 1.
        self.bufshr_subgrp_size = tuple(sgs if self.data_size(bl, dce) else 1
                                        for dce, sgs
                                        in enumerate(self.bufshr_subgrp_size))

        if all(sgs == 1 for sgs in self.bufshr_subgrp_size):
            return

        ## Loop structure.

        # The blocking factors and loop order that are related to BS.
        t_x = self.bl_ts[blp1]
        ord_x = self.bl_ords[blp1]

        # Non-trivial loops.
        nt_loops = set(lpe for lpe in range(le.NUM) if t_x[lpe] > 1)

        # To keep fetch times to all hierarchies unchanged, we fix some loops
        # without further blocking them in BS. See _set_fetch(), the
        # (unrelated) loops inside the innermost non-trivial dim loop does not
        # contribute to the fetch times, so we fix these loops for all data
        # categories.
        o_inntdim_loop = max(
            (self._innt_dim_loop(dce, t_x, ord_x) for dce in range(de.NUM)),
            key=lambda lpe: (ord_x[lpe] if lpe is not None else -1))
        # A tuple in the order of outer to inner, i.e., sort by inverse order.
        fixed_loops = tuple(sorted(
            (lpe for lpe in nt_loops if ord_x[lpe] < ord_x[o_inntdim_loop]),
            key=lambda lpe: ord_x[lpe],
            reverse=True))

        # The loops that can be further blocked without affecting the fetch
        # times to all hierarchies.
        flex_loops = nt_loops.difference(fixed_loops)

        ## Subgroup size candidates.

        def _min_subgrp_size(*dce_list):
            '''
            Get the minimum BS subgroup size, but not changing the current
            subgroup size. Minimize in the order of the given `dce_list`.
            '''
            # No duplication.
            assert len(dce_list) == len(set(dce_list))

            # Free capacity in each node's GBUF.
            free_cap = resource.size_gbuf - self.data_size(bl)

            sgs_list = list(self.bufshr_subgrp_size)

            for dce in dce_list:
                # Skip no sharing case.
                if sgs_list[dce] <= 1:
                    continue

                cur_dsz = self.data_size(bl, dce)
                tot_dsz = cur_dsz * self.bufshr_subgrp_size[dce]
                assert cur_dsz > 0 and tot_dsz > 0

                # min. sgs
                # s.t. tot_dsz / sgs <= free_cap + cur_dsz.
                for sgs in range(sgs_list[dce], 0, -1):
                    if self.bufshr_grp_size[dce] % sgs != 0:
                        # Require subgroup size to be a factor of the group
                        # size.
                        continue
                    if util.idivc(tot_dsz, sgs) <= free_cap + cur_dsz:
                        sgs_list[dce] = sgs
                    else:
                        break

                # Reduce free capacity.
                free_cap -= util.idivc(tot_dsz, sgs_list[dce]) - cur_dsz
                assert free_cap >= 0

            return tuple(sgs_list)

        # Original subgroup size.
        subgrp_size_cands = [self.bufshr_subgrp_size]
        # Reduce subgroup size if data can fit in fewer nodes. Consider all
        # orders about which data first shrink.
        subgrp_size_cands += set(_min_subgrp_size(*dce_list) for dce_list
                                 in itertools.permutations(range(de.NUM)))

        ## Sweep all BS schemes.

        def _sweep_bufshr():
            for subgrp_size in subgrp_size_cands:

                # `flex_loops` can be further blocked in BS, while others
                # cannot (set to 1).
                t_bs_tot = [t_x[lpe] if lpe in flex_loops else 1
                            for lpe in range(le.NUM)]

                for t_bs_frac in itertools.product(
                        *[util.factorize(t, 2) for t in t_bs_tot]):
                    t_bs = tuple(t[0] for t in t_bs_frac)

                    loops_bs_trivial = tuple(lpe for lpe in flex_loops
                                             if t_bs[lpe] == 1)

                    for loops_bs_nontrivial, loops_bot in itertools.product(
                            itertools.permutations([lpe for lpe in flex_loops
                                                    if t_bs[lpe] > 1]),
                            itertools.permutations(flex_loops)):

                        loops_bs = loops_bs_trivial + loops_bs_nontrivial

                        yield subgrp_size, t_bs, loops_bs, loops_bot

        ## BS NoC fetch times.

        dim_loops = [self.nld.data_loops[dce].loops() for dce in range(de.NUM)]

        def _is_dim_loop(lpe, dce, _dim_loops=dim_loops):
            return lpe in _dim_loops[dce]

        def _calc_bufshr_fetch(subgrp_size, t_bs, loops_bs, loops_bot):
            '''
            Calculate the BS scheme NoC fetch times. Return rotation fetch,
            wide fetch, and other statistics.

            `subgrp_size` is the BS subgroup size for each data category.

            `t_bs` is the blocking factors indexed by LoopEnum for the
            additional BS level between DRAM and GBUF, i.e., above `blp1`. They
            are fractorized from `t_x`. Only those in `flex_loops` can have
            non-1 values.

            `loops_bs` and `loops_bot` are ordered tuples of `flex_loops` from
            outer to inner, for the additional BS level and the original GBUF
            level (at the bottom) respectively.
            '''
            assert set(loops_bs) == set(loops_bot) == flex_loops
            assert all(b <= x for b, x in zip(t_bs, t_x))
            assert all(t_bs[lpe] == 1 or lpe in flex_loops
                       for lpe in range(le.NUM))

            # Make a list of tuples (LoopEnum, blocking factor)`, each
            # corresponds to a non-trivial loop in the additional BS level and
            # the original GBUF level, ordered from outer to inner.
            lp_t_list = []
            # Additional BS level.
            lp_t_list += [(lpe, t_bs[lpe])
                          for lpe in loops_bs if t_bs[lpe] > 1]
            # GBUF level flex loops.
            lp_t_list += [(lpe, util.idivc(t_x[lpe], t_bs[lpe]))
                          for lpe in loops_bot if t_x[lpe] > t_bs[lpe]]
            # GBUF level fixed loops.
            lp_t_list += [(lpe, t_x[lpe]) for lpe in fixed_loops]
            # Check.
            assert all(tpl[1] > 1 for tpl in lp_t_list)

            # Total rotation rounds (over all GBUF filling).
            rot_rnd_cnts = []
            # Number of rotation units.
            rot_unit_cnts = []
            # Wide fetch widths.
            wide_fetch_widths = []

            # Rotation NoC fetch times.
            rot_fetch = []
            # Wide fetch NoC fetch times.
            wide_fetch = []

            for dce in range(de.NUM):

                buf_fetch = self.fetch[blp1][dce]
                mem_fetch = self.fetch[blp1-1][dce]

                # Index of the outermost dim loop in `lp_t_list`. None if all
                # dim loops are trivial.
                idx_odlp = next((i for i, tpl in enumerate(lp_t_list)
                                 if _is_dim_loop(tpl[0], dce)),
                                None)

                # Rotation rounds.
                rotrnds = 1
                if idx_odlp is None or subgrp_size[dce] == 1:
                    # No rotation.
                    rotrnds = 0
                elif idx_odlp is not None:
                    # All unrelated loop factors above the outermost dim loop.
                    # At DRAM level.
                    rotrnds *= util.prod(self.nld.data_loops[dce]
                                         .drop(self._bl_tp(slice(blp1))))
                    # At GBUF level.
                    rotrnds *= util.prod(tpl[1] for tpl
                                         in itertools.islice(lp_t_list,
                                                             idx_odlp))
                    assert ((buf_fetch + 1) // 2 if dce == de.OFM
                            else buf_fetch) % rotrnds == 0
                    assert rotrnds % ((mem_fetch + 1) // 2 if dce == de.OFM
                                      else mem_fetch) == 0
                # Optimization: after fetching data into GBUF, if the data only
                # rotate a single time before being replaced, we do not need to
                # store them after this single use. So instead we can stream
                # each rotation unit to all the nodes, and replace it by the
                # next rotation unit one by one. This is already supported as
                # the data will be broadcast to all nodes regardless of who
                # stores it (see partition).
                if rotrnds == ((mem_fetch + 1) // 2 if dce == de.OFM
                               else mem_fetch):
                    rotrnds = 0
                rot_rnd_cnts.append(rotrnds)

                # Number of rotation units.
                rotunits = 1
                # All dimension sizes of the outermost adjacent dim loops.
                if idx_odlp is not None:
                    rotunits = util.prod(tpl[1] for tpl
                                         in itertools.takewhile(
                                             lambda tpl, dce_=dce:
                                             _is_dim_loop(tpl[0], dce_),
                                             itertools.islice(lp_t_list,
                                                              idx_odlp, None)))
                rot_unit_cnts.append(rotunits)

                # Wide fetch width.
                wf_width = 1. * subgrp_size[dce] / rotunits
                wide_fetch_widths.append(wf_width)

                # Wide fetch times.
                wf_per_bufacc = bufshr.nhops_wide_fetch_once(
                    dce, subgrp_size[dce], wf_width)
                # Use REGF filling (GBUF fetch).
                # The last wide fetch before rotation can be combined with the
                # rotation steps.
                if dce == de.OFM:
                    # For OFM, if we do multiple wide fetch per rotation step,
                    # the last one has both read and write. If there is only
                    # one wide fetch per rotation step, it only has write.
                    if buf_fetch > 2 * rotrnds - 1:
                        comb_wf_fetch = 2 * rotrnds
                    else:
                        assert buf_fetch == 2 * rotrnds - 1
                        comb_wf_fetch = 2 * rotrnds - 1
                else:
                    comb_wf_fetch = rotrnds
                # Since we do not rotate the last step, when wide fetch is
                # non-0 (i.e., the last rotation unit is larger than one node
                # buffer size), the wide fetch of the last unit has no rotation
                # to combine with.
                comb_wf_fetch *= 1. * (rotunits - 1) / rotunits
                wf = wf_per_bufacc * (buf_fetch - comb_wf_fetch)
                assert wf > -1e-4
                wide_fetch.append(wf)

                # Rotation fetch times.
                rf_per_rot = bufshr.nhops_rotate_all(
                    dce, subgrp_size[dce], rotunits)
                rf = rf_per_rot * rotrnds
                rot_fetch.append(rf)

            return rot_fetch, wide_fetch, \
                    rot_rnd_cnts, rot_unit_cnts, wide_fetch_widths

        ## Search for the best BS scheme.

        def _key_func(tuple_):
            rot_fetch, wide_fetch = _calc_bufshr_fetch(*tuple_)[:2]
            return sum(self._calc_bufshr_rotation_access(rot_fetch)) \
                    + sum(self._calc_bufshr_widefetch_access(wide_fetch))
        subgrp_size, t_bs, loops_bs, loops_bot = \
                min(_sweep_bufshr(), key=_key_func)

        # Subgroup size.
        self.bufshr_subgrp_size = subgrp_size

        # Loop blocking factors and order.
        new_ord = [-1] * le.NUM
        ord_idx = 0
        for lpe in reversed(loops_bot + fixed_loops):
            new_ord[lpe] = ord_idx
            ord_idx += 1
        for lpe in range(le.NUM):
            if new_ord[lpe] < 0:
                new_ord[lpe] = ord_idx
                ord_idx += 1
        self.bl_ords[blp1] = tuple(new_ord)

        # Additional BS level.
        new_ord_bs = [-1] * le.NUM
        ord_idx = 0
        for lpe in reversed(loops_bs):
            if t_bs[lpe] > 1:
                new_ord_bs[lpe] = ord_idx
                ord_idx += 1
        for lpe in range(le.NUM):
            if new_ord_bs[lpe] < 0:
                new_ord_bs[lpe] = ord_idx
                ord_idx += 1
        self.bufshr_bs_t = tuple(t_bs)
        self.bufshr_bs_ord = tuple(new_ord_bs)

        # Set stats.
        self.bufshr_rot_fetch, self.bufshr_wide_fetch, \
                self.bufshr_rot_round_cnt, self.bufshr_rot_unit_cnt, \
                self.bufshr_wide_fetch_width = \
                _calc_bufshr_fetch(subgrp_size, t_bs, loops_bs, loops_bot)

    def _calc_bufshr_rotation_access(self, bufshr_rot_fetch):
        ''' Calculate the BS rotation NoC accesses, over all nodes. '''
        # All-node access needs to multiply number of groups.
        return [self.total_access_gbuf[dce]
                * bufshr_rot_fetch[dce]
                * (self.num_nodes // self.bufshr_grp_size[dce])
                for dce in range(de.NUM)]

    def _calc_bufshr_widefetch_access(self, bufshr_wide_fetch):
        ''' Calculate the BS wide fetch NoC accesses, over all nodes. '''
        # All-node access needs to multiply number of groups.
        return [self.total_access_gbuf[dce]
                * bufshr_wide_fetch[dce]
                * (self.num_nodes // self.bufshr_grp_size[dce])
                for dce in range(de.NUM)]

