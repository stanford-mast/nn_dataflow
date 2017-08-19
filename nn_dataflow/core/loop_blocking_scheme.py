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

from . import data_category_enum as de
from . import loop_enum as le
from . import mem_hier_enum as me
from .. import util

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

    def __init__(self, nested_loop_desc, bl_ts, bl_ords, resource, part_occ,
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

        `part_occ` is the partitioning occupation.
        '''

        # pylint: disable=invalid-name
        BL = self.BL

        # Loop structure.
        self.nld = nested_loop_desc

        # Check lengths and values.
        assert len(bl_ts) == BL.NUM + 1, \
                'LoopBlockingScheme: bl_ts has invalid length.'
        assert all(len(bl_t) == le.NUM for bl_t in bl_ts), \
                'LoopBlockingScheme: bl_ts elements have invalid length.'
        assert len(bl_ords) == BL.NUM, \
                'LoopBlockingScheme: bl_ords has invalid length.'
        assert all(sorted(bl_ord) == range(le.NUM) for bl_ord in bl_ords), \
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

        # Parallel partitioning.
        self.num_nodes = resource.proc_region.dim.size()
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

        lp_ts = zip(*self.bl_ts)

        return OrderedDict([('cost', self.get_cost(cost)),
                            ('ops', self.ops),
                            ('time', self.time),
                            ('access', self.access),
                            ('fetch', self.fetch),
                            ('size', size),
                            ('unit_size', self.unit_size),
                            ('unit_cnt', self.unit_cnt),
                            ('part_occ', self.part_occ),
                            ('ti', tuple(lp_ts[le.IFM])),
                            ('to', tuple(lp_ts[le.OFM])),
                            ('tb', tuple(lp_ts[le.BAT])),
                            ('orders', self.bl_ords),
                           ])

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

        # Between GBUF and REGF.
        t_x = self.bl_ts[bl_regf]
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

        self.ops = self.nld.unit_ops * self.lcnt * self.num_nodes \
                * self.part_occ
        self.time = self.nld.unit_time * self.lcnt

        self.access[me.REGF] = [v * self.lcnt * t
                                * self.num_nodes * self.part_occ for v, t
                                in zip(self.nld.unit_access[me.REGF],
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
                                for dce in range(de.NUM)]

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
        return [self.nld.data_loops[dce].data_cnt(bl_t)
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
            gens[rev_order[lpe]] = xrange(t_x[lpe])

        for idx in itertools.product(*gens):
            # Index now is in the loop order from outer to inner. Reorder to be
            # in LoopEnum order.
            yield tuple(idx[rev_order[lpe]] for lpe in range(le.NUM))

