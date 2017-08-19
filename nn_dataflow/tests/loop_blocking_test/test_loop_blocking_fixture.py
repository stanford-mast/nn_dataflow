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

import itertools
import unittest

from nn_dataflow.core import ConvLayer, PoolingLayer
from nn_dataflow.core import Cost
from nn_dataflow.core import DataDimLoops
from nn_dataflow.core import DataCategoryEnum as de
from nn_dataflow.core import LoopBlockingScheme
from nn_dataflow.core import LoopEnum as le
from nn_dataflow.core import MapStrategyEyeriss
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow.core import NestedLoopDesc
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import Option
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource
from nn_dataflow import util

class TestLoopBlockingFixture(unittest.TestCase):
    ''' Base fixture class for LoopBlocking tests. '''

    def setUp(self):

        # Workload.
        self.layer = {}
        self.layer['BASE'] = ConvLayer(12, 10, 28, 3)
        self.layer['LGFIL'] = ConvLayer(2, 4, 28, 20)
        self.layer['POOL'] = PoolingLayer(32, 28, 2)
        self.batch_size = 4

        # Resource.
        self.resource = {}
        dim_array = PhyDim2(16, 16)
        proc_region = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                 type=NodeRegion.PROC)
        data_regions = (NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                   type=NodeRegion.DATA),)
        # Typical resource.
        self.resource['BASE'] = Resource(
            proc_region=proc_region, data_regions=data_regions,
            dim_array=dim_array, size_gbuf=65536, size_regf=64)
        # Larger resource with sufficient capacity, to make all schemes valid.
        self.resource['LG'] = Resource(
            proc_region=proc_region, data_regions=data_regions,
            dim_array=dim_array, size_gbuf=1024 ** 3, size_regf=1024 ** 3)
        # Small resource.
        self.resource['SM'] = Resource(
            proc_region=proc_region, data_regions=data_regions,
            dim_array=dim_array, size_gbuf=4096, size_regf=16)

        # Nested loop description after mapping.
        self.nld = {}
        self.nld['BASE'] = next(MapStrategyEyeriss(self.layer['BASE'],
                                                   self.batch_size, dim_array)
                                .gen_nested_loop_desc())
        self.nld['LGFIL'] = next(MapStrategyEyeriss(self.layer['LGFIL'],
                                                    self.batch_size, dim_array)
                                 .gen_nested_loop_desc())
        self.nld['POOL'] = next(MapStrategyEyeriss(self.layer['POOL'],
                                                   self.batch_size, dim_array)
                                .gen_nested_loop_desc())
        # Fake nested loop, with zero filter size.
        self.nld['ZERO_FIL'] = NestedLoopDesc(loopcnt=(12, 10, 4),
                                              usize_gbuf=(0, 1000, 800),
                                              usize_regf=(0, 3, 1),
                                              unit_access=((0, 1000, 800),
                                                           (0, 1000, 800),
                                                           (3, 9, 7),
                                                           (1, 1, 1)),
                                              data_loops=(DataDimLoops(le.IFM,
                                                                       le.OFM),
                                                          DataDimLoops(le.IFM,
                                                                       le.BAT),
                                                          DataDimLoops(le.OFM,
                                                                       le.BAT)),
                                              unit_ops=1, unit_time=1)
        # Fake nested loop, with zero ifmap size.
        self.nld['ZERO_IFM'] = NestedLoopDesc(loopcnt=(12, 10, 4),
                                              usize_gbuf=(9, 0, 800),
                                              usize_regf=(3, 0, 1),
                                              unit_access=((9, 0, 800),
                                                           (9, 0, 800),
                                                           (3, 9, 7),
                                                           (1, 1, 1)),
                                              data_loops=(DataDimLoops(le.IFM,
                                                                       le.OFM),
                                                          DataDimLoops(le.IFM,
                                                                       le.BAT),
                                                          DataDimLoops(le.OFM,
                                                                       le.BAT)),
                                              unit_ops=1, unit_time=1)

        # Options.
        self.options = {}
        # Basic.
        self.options['BASE'] = Option(ntops=2 ** 30)
        # Multiprocessing.
        self.options['MP'] = Option(ntops=2 ** 30, nprocesses=8)
        # Limited top schemes.
        self.options['NTOPS'] = Option(ntops=10)
        # Bypass.
        self.options['BYP'] = Option(sw_gbuf_bypass=(True,) * 3, ntops=2 ** 30)
        # Bypass solver.
        self.options['BYPSOL'] = Option(sw_gbuf_bypass=(True,) * 3,
                                        sw_solve_loopblocking=True,
                                        ntops=2 ** 30)

        # Cost.
        self.cost = Cost(mac_op=1, mem_hier=(200, 6, 2, 1),
                         noc_hop=50, unit_static=50)

        # Partition occupation.
        self.part_occ = 0.91


    def _lbs(self, bl_ts, bl_ords=None, wlkey='BASE', rsrckey='BASE',
             optkey='BASE', part_occ=1):
        ''' Make a LoopBlockingScheme instance. '''
        bl_ords = (tuple(range(le.NUM)), tuple(range(le.NUM))) \
                if not bl_ords else bl_ords
        return LoopBlockingScheme(self.nld[wlkey], bl_ts, bl_ords,
                                  self.resource[rsrckey], part_occ,
                                  self.options[optkey])

    def _gen_loopblocking_all(self, wlkey='BASE'):
        ''' Generate all combinations of loop blocking factors and orders. '''
        for ti, to, tb, orders in itertools.product(
                util.factorize(self.nld[wlkey].loopcnt[le.IFM], 3),
                util.factorize(self.nld[wlkey].loopcnt[le.OFM], 3),
                util.factorize(self.nld[wlkey].loopcnt[le.BAT], 3),
                itertools.product(
                    itertools.permutations(range(le.NUM)),
                    itertools.permutations(range(le.NUM)))):
            lp_ts = [None] * le.NUM
            lp_ts[le.IFM] = ti
            lp_ts[le.OFM] = to
            lp_ts[le.BAT] = tb
            yield tuple(zip(*lp_ts)), orders

    def _make_bl_ts(self, ti_part, to_part, tb_part, wlkey='BASE'):
        '''
        Make a set of blocking factors. `ti_part`, `to_part`, `tb_part` can
        contain one 0 value to be filled.
        '''
        try:
            idx = ti_part.index(0)
        except ValueError:
            ti = ti_part
        else:
            ti = [ti_part[x] if x != idx
                  else util.idivc(self.nld[wlkey].loopcnt[le.IFM],
                                  util.prod(ti_part[:idx] + ti_part[idx+1:]))
                  for x in range(3)]
        try:
            idx = to_part.index(0)
        except ValueError:
            to = to_part
        else:
            to = [to_part[x] if x != idx
                  else util.idivc(self.nld[wlkey].loopcnt[le.OFM],
                                  util.prod(to_part[:idx] + to_part[idx+1:]))
                  for x in range(3)]
        try:
            idx = tb_part.index(0)
        except ValueError:
            tb = tb_part
        else:
            tb = [tb_part[x] if x != idx
                  else util.idivc(self.nld[wlkey].loopcnt[le.BAT],
                                  util.prod(tb_part[:idx] + tb_part[idx+1:]))
                  for x in range(3)]
        lp_ts = [None] * le.NUM
        lp_ts[le.IFM] = ti
        lp_ts[le.OFM] = to
        lp_ts[le.BAT] = tb
        return tuple(zip(*lp_ts))


    class _SimBuffer(object):
        ''' A data buffer model for simulation. '''

        def __init__(self, dce, buf_cnt_pr, unit_size, bypass=False):

            self.dce = dce
            self.bypass = bypass

            # Accesses to this level, in unit counts (* unit size).
            self.access = 0

            # The size of one unit.
            self.unit_size = unit_size

            if self.bypass:
                return

            # The buffered data range, in the form of the range index, of all
            # dimensions. E.g., (ri0, ri1).
            self.data = (float('nan'), float('nan'))

            # The count of buffered units, aka, range size, of all dimensions.
            # E.g., (c0, c1).
            self.buf_cnt_pr = buf_cnt_pr

        def access_size(self):
            ''' Get access size. '''
            return self.access * self.unit_size

        def do_access(self, idx_pr, cnt_pr, read=1, write=0):
            '''
            Access the buffer by `read` and/or `write`, with the unit index
            `idx_pr` and count `cnt_pr`, of all dimensions.

            Return the count of the accessing data to the next level, of all
            dimensions.
            '''
            if self.bypass:
                # Bypass, relay to the next level.
                return cnt_pr

            # Range index.
            ridx_pr = tuple(idx // buf_cnt for idx, buf_cnt
                            in zip(idx_pr, self.buf_cnt_pr))

            # Access.
            self.access += util.prod(cnt_pr) * (read + write)

            if ridx_pr == self.data:
                # Hit.
                return (0, 0)

            # Miss.
            self.data = ridx_pr
            return self.buf_cnt_pr

    def _sim_access_conv(self, lbs):
        '''
        Get data access by actually simulating and generating loops for CONV
        layer.
        '''
        self.assertTrue(lbs.is_valid(), '_sim_access_conv: invalid lbs.')

        data_loops = lbs.nld.data_loops

        lpts = zip(*lbs.bl_ts)

        # Get buffered unit counts at each level.
        dram_buf_cnt_pr_list = [tuple(util.prod(lpts[lpe])
                                      for lpe in data_loops[dce].loops())
                                for dce in range(de.NUM)]
        gbuf_buf_cnt_pr_list = [tuple(util.prod(lpts[lpe][1:])
                                      for lpe in data_loops[dce].loops())
                                for dce in range(de.NUM)]
        regf_buf_cnt_pr_list = [tuple(util.prod(lpts[lpe][2:])
                                      for lpe in data_loops[dce].loops())
                                for dce in range(de.NUM)]

        # Initialize SimBuffer.
        drams = [None] * de.NUM
        for dce, buf_cnt_pr in enumerate(dram_buf_cnt_pr_list):
            drams[dce] = self._SimBuffer(dce, buf_cnt_pr,
                                         lbs.nld.unit_access[me.DRAM][dce]
                                         if lbs.stored_in_gbuf[dce]
                                         else lbs.nld.unit_access[me.GBUF][dce],
                                        )
        gbufs = [None] * de.NUM
        for dce, buf_cnt_pr in enumerate(gbuf_buf_cnt_pr_list):
            gbufs[dce] = self._SimBuffer(dce, buf_cnt_pr,
                                         lbs.nld.unit_access[me.GBUF][dce],
                                         bypass=(not lbs.stored_in_gbuf[dce]),
                                        )
        regfs = [None] * de.NUM
        for dce, buf_cnt_pr in enumerate(regf_buf_cnt_pr_list):
            regfs[dce] = self._SimBuffer(dce, buf_cnt_pr,
                                         lbs.nld.unit_access[me.REGF][dce],
                                        )

        # Already generated psum for OFM.
        ofm_psum = set()

        # Simulation.
        for idx_tuple in lbs.gen_index():

            for dce in range(de.NUM):

                idx_pr = tuple(data_loops[dce].take(idx_tuple))

                if dce == de.OFM:
                    # Fetch and writeback, unless for the first time (no fetch).
                    write = 1
                    read = 1 if idx_pr in ofm_psum else 0
                    ofm_psum.add(idx_pr)
                else:
                    read = 1
                    write = 0

                # PE.
                cnt_pr = (1, 1)

                # REGF.
                cnt_pr = regfs[dce].do_access(idx_pr, cnt_pr, read, write)
                if not any(cnt_pr):
                    continue

                # GBUF.
                cnt_pr = gbufs[dce].do_access(idx_pr, cnt_pr, read, write)
                if not any(cnt_pr):
                    continue

                # DRAM.
                cnt_pr = drams[dce].do_access(idx_pr, cnt_pr, read, write)
                if not any(cnt_pr):
                    continue

        dram_access = [drams[dce].access_size() for dce in range(de.NUM)]
        gbuf_access = [gbufs[dce].access_size() for dce in range(de.NUM)]
        return dram_access, gbuf_access


    def _regularized_scheme(self, bl_ts, bl_ords):
        ''' Get the regularized scheme which will not be skipped. '''

        assert isinstance(bl_ts, tuple) and isinstance(bl_ords, tuple)
        assert all(isinstance(t, tuple) for t in bl_ts)
        assert all(isinstance(o, tuple) for o in bl_ords)

        reg_lpts = [[] for _ in range(le.NUM)]
        reg_ords = tuple()

        outer_level_innermost_loop = None

        for t_, ord_ in itertools.izip_longest(bl_ts, bl_ords, fillvalue=None):

            # Non-trivial loops and trivial loops of this level.
            ntlp_list = sorted(lpe for lpe in range(le.NUM)
                               if t_[lpe] > 1)
            trlp_list = sorted(lpe for lpe in range(le.NUM)
                               if lpe not in ntlp_list)

            # Innermost non-trivial loop.
            try:
                ntlp_innermost = min(ntlp_list,
                                     key=lambda lpe, o=ord_: o[lpe])
            except (ValueError, TypeError):
                # All trivial loops or no order (last level).
                assert not ntlp_list or not ord_
                ntlp_innermost = None

            if ord_:
                # Order trivial and non-trivial loops separately of this level.
                reg_ord = [None] * le.NUM
                # Innermost loop.
                try:
                    reg_ord[ntlp_innermost] = 0
                    o = 1
                except TypeError:
                    o = 0
                # First non-trivial loops (inner), then trivial loops (outer).
                for lpe in ntlp_list + trlp_list:
                    if lpe == ntlp_innermost:
                        continue
                    reg_ord[lpe] = o
                    o += 1
                assert o == le.NUM

                # Loop orders.
                reg_ords += (tuple(reg_ord),)

            # Blocking factors.
            for lpe in range(le.NUM):
                reg_lpts[lpe].append(t_[lpe])

            if ntlp_list:
                if outer_level_innermost_loop != ntlp_innermost \
                        and outer_level_innermost_loop in ntlp_list:
                    # Adjust blocking factors by merging two adjacent loops to
                    # the outer one.
                    lpe = outer_level_innermost_loop
                    reg_lpts[lpe][-2] *= reg_lpts[lpe][-1]
                    reg_lpts[lpe][-1] = 1

                outer_level_innermost_loop = ntlp_innermost

        reg_ts = tuple(zip(*reg_lpts))

        if reg_ts == bl_ts and reg_ords == bl_ords:
            return reg_ts, reg_ords

        # Recursive call, since loop merging/reordering may cause further loop
        # merging/reordering.
        return self._regularized_scheme(reg_ts, reg_ords)

