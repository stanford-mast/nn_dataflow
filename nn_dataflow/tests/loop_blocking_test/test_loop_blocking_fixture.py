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
import unittest

from nn_dataflow.core import partition
from nn_dataflow.core import BufShrScheme
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
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource
from nn_dataflow.core import SchedulingConstraint
from nn_dataflow import util

class TestLoopBlockingFixture(unittest.TestCase):
    ''' Base fixture class for LoopBlocking tests. '''
    # pylint: disable=too-many-instance-attributes

    def setUp(self):

        # Workload.
        self.layer = {}
        self.layer['BASE'] = ConvLayer(12, 10, 28, 3)
        self.layer['LGFIL'] = ConvLayer(2, 4, 28, 20)
        self.layer['POOL'] = PoolingLayer(32, 28, 2)
        self.layer['PAR'] = ConvLayer(24, 36, 56, 3)
        self.batch_size = 4

        # Resource.
        self.resource = {}
        dim_array = PhyDim2(16, 16)
        proc_region = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                 type=NodeRegion.PROC)
        data_region = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                 type=NodeRegion.DRAM)
        # Typical resource.
        self.resource['BASE'] = Resource(
            proc_region=proc_region, dram_region=data_region,
            src_data_region=data_region, dst_data_region=data_region,
            dim_array=dim_array, size_gbuf=65536, size_regf=64,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)
        # Larger resource with sufficient capacity, to make all schemes valid.
        self.resource['LG'] = Resource(
            proc_region=proc_region, dram_region=data_region,
            src_data_region=data_region, dst_data_region=data_region,
            dim_array=dim_array, size_gbuf=1024 ** 3, size_regf=1024 ** 3,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)
        # Small resource.
        self.resource['SM'] = Resource(
            proc_region=proc_region, dram_region=data_region,
            src_data_region=data_region, dst_data_region=data_region,
            dim_array=dim_array, size_gbuf=4096, size_regf=16,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)
        # Multi-node parallel resource.
        self.resource['PAR'] = Resource(
            proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                   dim=PhyDim2(4, 2),
                                   type=NodeRegion.PROC),
            dram_region=data_region,
            src_data_region=data_region, dst_data_region=data_region,
            dim_array=dim_array, size_gbuf=25000, size_regf=64,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)
        # Resource with no data regions.
        proc_data_region = NodeRegion(origin=PhyDim2(1, 1), dim=PhyDim2(1, 1),
                                      type=NodeRegion.PROC)
        self.resource['SRCNOTDATA'] = Resource(
            proc_region=proc_region, dram_region=data_region,
            src_data_region=proc_data_region, dst_data_region=data_region,
            dim_array=dim_array, size_gbuf=1024 ** 3, size_regf=1024 ** 3,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)
        self.resource['DSTNOTDATA'] = Resource(
            proc_region=proc_region, dram_region=data_region,
            src_data_region=data_region, dst_data_region=proc_data_region,
            dim_array=dim_array, size_gbuf=1024 ** 3, size_regf=1024 ** 3,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)
        self.resource['DATALOCAL'] = Resource(
            proc_region=proc_region, dram_region=data_region,
            src_data_region=proc_region, dst_data_region=proc_region,
            dim_array=dim_array, size_gbuf=1024 ** 3, size_regf=1024 ** 3,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)
        # Filter pinning.
        self.resource['FILPIN'] = Resource(
            proc_region=proc_region, dram_region=data_region,
            src_data_region=data_region, dst_data_region=data_region,
            dim_array=dim_array, size_gbuf=1024 ** 3, size_regf=1024 ** 3,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=True)

        # Nested loop description after mapping.
        self.nld = {}
        self.nld['BASE'] = next(MapStrategyEyeriss(self.layer['BASE'],
                                                   self.batch_size, 1,
                                                   dim_array)
                                .gen_nested_loop_desc())
        self.nld['LGFIL'] = next(MapStrategyEyeriss(self.layer['LGFIL'],
                                                    self.batch_size, 1,
                                                    dim_array)
                                 .gen_nested_loop_desc())
        self.nld['POOL'] = next(MapStrategyEyeriss(self.layer['POOL'],
                                                   self.batch_size, 1,
                                                   dim_array)
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

        # Fake partition scheme.
        self.part = PartitionScheme(range(pe.NUM), ((1, 1),) * pe.NUM)

        # Fake buffer sharing scheme.
        self.bufshr = BufShrScheme(proc_region, self.part)

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
        # Access forwarding.
        self.options['ACCFWD'] = Option(hw_access_forwarding=True,
                                        ntops=2 ** 30)
        # Buffer sharing.
        self.options['BUFSHR'] = Option(hw_gbuf_sharing=True,
                                        ntops=2 ** 30)
        # Buffer sharing with bypassing.
        self.options['BUFSHR-BYP'] = Option(sw_gbuf_bypass=(True,) * 3,
                                            hw_gbuf_sharing=True,
                                            ntops=2 ** 30)

        # Constraint.
        self.none_cstr = SchedulingConstraint()
        self.cstr = SchedulingConstraint(topifm=1, topbat=1)

        # Cost.
        self.cost = Cost(mac_op=1, mem_hier=(200, 6, 2, 1),
                         noc_hop=50, idl_unit=50)


    def _lbs(self, bl_ts, bl_ords=None, wlkey='BASE', rsrckey='BASE',
             optkey='BASE'):
        ''' Make a LoopBlockingScheme instance. '''
        bl_ords = (tuple(range(le.NUM)), tuple(range(le.NUM))) \
                if not bl_ords else bl_ords
        return LoopBlockingScheme(self.nld[wlkey], bl_ts, bl_ords,
                                  self.resource[rsrckey], self.bufshr,
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

    def _part_nld(self, part, layerkey='PAR'):
        ''' Make a partitioned NestedLoopDesc and its partition occupation. '''
        p_layer, p_batch_size, p_occ = part.part_layer(self.layer[layerkey],
                                                       self.batch_size)
        p_nld = next(MapStrategyEyeriss(p_layer, p_batch_size, p_occ,
                                        self.resource['PAR'].dim_array)
                     .gen_nested_loop_desc())
        return p_nld

    def _gen_all_partition(self, layerkey='PAR'):
        '''
        Generate PartitionScheme.
        '''
        options = Option(partition_hybrid=True,
                         partition_batch=True,
                         partition_ifmaps=True,
                         ntops=2 ** 30)

        for part in partition.gen_partition(
                self.layer[layerkey], self.batch_size,
                self.resource['PAR'].proc_region.dim, options):
            yield part

    def _total_part_size(self, part, layerkey='PAR'):
        ''' Get the total partitioned data size. '''
        layer = self.layer[layerkey]

        nifm = util.idivc(layer.nifm, part.size(pe.INPP)) * part.size(pe.INPP)
        nofm = util.idivc(layer.nofm, part.size(pe.OUTP)) * part.size(pe.OUTP)
        hofm = util.idivc(layer.hofm, part.dim(pe.OFMP).h) * part.dim(pe.OFMP).h
        wofm = util.idivc(layer.wofm, part.dim(pe.OFMP).w) * part.dim(pe.OFMP).w
        batch_size = util.idivc(self.batch_size, part.size(pe.BATP)) \
                * part.size(pe.BATP)

        full_layer = ConvLayer(nifm, nofm, (hofm, wofm),
                               (layer.hfil, layer.wfil),
                               (layer.htrd, layer.wtrd))
        filter_size = full_layer.total_filter_size()
        ifmap_size = full_layer.total_ifmap_size(batch_size)
        ofmap_size = full_layer.total_ofmap_size(batch_size)

        self.assertGreaterEqual(filter_size, layer.total_filter_size())
        self.assertLess(filter_size, layer.total_filter_size() * 1.2 * 1.2)
        self.assertGreaterEqual(ofmap_size,
                                layer.total_ofmap_size(self.batch_size))
        self.assertLess(ofmap_size,
                        layer.total_ofmap_size(self.batch_size)
                        * 1.2 * 1.2 * 1.2)
        self.assertGreaterEqual(ifmap_size,
                                layer.total_ifmap_size(self.batch_size))

        return filter_size, ifmap_size, ofmap_size

    def _bufshr_params(self, lbs):
        '''
        Get buffer sharing parameters.

        Return subgroup sizes, rotation unit counts.

        Finally, a list of ordered loops as a tuple of LoopEnum and blocking
        factor ordered from outermost to innermost excluding trivial loops.
        '''
        # GBUF level.
        blp1 = lbs.BL.GBUF + 1
        t_x = lbs.bl_ts[blp1]
        ord_x = lbs.bl_ords[blp1]
        # BS level.
        t_bs = lbs.bufshr_bs_t
        ord_bs = lbs.bufshr_bs_ord

        self.assertTrue(all(x % b == 0 for x, b in zip(t_x, t_bs)))

        subgrp_size = lbs.bufshr_subgrp_size
        rot_unit_cnt = lbs.bufshr_rot_unit_cnt

        # Loops as a tuple of LoopEnum and blocking factor, ordered from
        # outermost to innermost, excluding trivial loops.
        lp_t_list = sorted([(lpe, t_bs[lpe])
                            for lpe in range(le.NUM) if t_bs[lpe] > 1],
                           key=lambda tpl: ord_bs[tpl[0]],
                           reverse=True) \
                  + sorted([(lpe, t_x[lpe] // t_bs[lpe])
                            for lpe in range(le.NUM) if t_x[lpe] > t_bs[lpe]],
                           key=lambda tpl: ord_x[tpl[0]],
                           reverse=True)

        return subgrp_size, rot_unit_cnt, lp_t_list


    class _SimBuffer():
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

            # Range index cache.
            self.ridx_pr_cache = {}

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
            ridx_pr = self._range_idx_pr(idx_pr)

            # Access.
            self.access += util.prod(cnt_pr) * (read + write)

            if ridx_pr == self.data:
                # Hit.
                return (0, 0)

            # Miss.
            self.data = ridx_pr
            return self.buf_cnt_pr

        def _range_idx_pr(self, idx_pr):
            ''' Get the range index of all dimensions. '''
            ridx_pr = self.ridx_pr_cache.get(idx_pr, None)
            if ridx_pr is None:
                ridx_pr = tuple(idx // buf_cnt for idx, buf_cnt
                                in zip(idx_pr, self.buf_cnt_pr))
                self.ridx_pr_cache[idx_pr] = ridx_pr
            return ridx_pr

    class _SimBufferSharing(_SimBuffer):
        ''' A data buffer model with buffer sharing. '''

        def __init__(self, dce, buf_cnt_pr, unit_size,
                     subgrp_size, rot_unit_cnt, lp_t_list, dim_loops,
                     bypass=False):

            # pylint: disable=protected-access
            self.base = super(TestLoopBlockingFixture._SimBufferSharing, self)

            self.base.__init__(dce, buf_cnt_pr, unit_size, bypass=bypass)

            # Number of rotation steps, of each range.
            self.rot_step_cnt = {}
            # Rotation accesses, in unit counts (* unit size).
            self.rot_access = 0
            # Wide fetch accesses, in unit counts (* unit size).
            self.wf_access = 0

            # Rotation rounds per load of a range. If only rotate a single
            # round per data load, the rotation is unnecessary.
            self.rot_rnd_cnt_per_load = None

            if self.bypass:
                return

            # Subrange.
            # A list in the accessing order of subrange indexes, i.e., the
            # ranges of the next level; and the unit counts in one subrange.
            self.subrng_list, self.subrng_cnt_pr = \
                    self._init_sub_range(lp_t_list, dim_loops)
            # Subrange index to the position in the list.
            self.subrng_idx_dict = \
                    dict((sr, i) for i, sr in enumerate(self.subrng_list))
            # Number of subranges.
            self.subrng_num = len(self.subrng_list)

            # Local buffer.
            self.buf_num = subgrp_size
            # Number of subranges in each buffer.
            self.buf_subrng_num = 1. * self.subrng_num / self.buf_num

            # The location centroid of each subrange, i.e., buffer index
            # weighted by fraction.
            self.buf_subrng_centroid = []
            cur_buf_cap = self.buf_subrng_num
            cur_buf_idx = 0
            for _ in range(self.subrng_num):
                centroid = 0
                rem_frac = 1.
                while rem_frac > 0.:
                    if cur_buf_cap >= rem_frac:
                        # Fits in the current buffer.
                        centroid += cur_buf_idx * rem_frac
                        cur_buf_cap -= rem_frac
                        rem_frac = 0.
                        break
                    # Partially fits.
                    centroid += cur_buf_idx * cur_buf_cap
                    rem_frac -= cur_buf_cap
                    cur_buf_cap = self.buf_subrng_num
                    cur_buf_idx += 1
                self.buf_subrng_centroid.append(centroid)

            # Rotation unit.
            # Rotation step happens when moving to the new rotation unit.
            assert self.subrng_num % rot_unit_cnt == 0
            self.rot_unit_size = self.subrng_num // rot_unit_cnt
            # Steps per rotation round.
            self.rot_steps_per_round = 1
            while (self.rot_steps_per_round * self.rot_unit_size
                   + self.buf_subrng_num < self.subrng_num
                   and (self.rot_steps_per_round + 1) * self.rot_unit_size
                   < self.subrng_num):
                self.rot_steps_per_round += 1

            # The rotation unit currently worked on.
            self.cur_rot_unit = 0
            # Rotation steps of the current load of the current range.
            self.cur_rot_step_cnt = 0

            # Last wide fetch subrange index.
            self.last_wf_subrng_idx = 0
            # Amount of sequential wide fetch, can be combined with rotation.
            self.seq_wf_acc = 0
            # Total saved (combined with rotation) wide fetch access.
            self.saved_wf_access = 0

            # Subrange index cache.
            self.sridx_pr_cache = {}

        def rotation_rounds(self):
            ''' Get number of rotation rounds. '''

            # Ensure all ranges have the same rotation steps.
            steps_list = tuple(self.rot_step_cnt.values())
            if not steps_list:
                return 0
            assert all(s == steps_list[0] for s in steps_list)
            steps = steps_list[0]
            if steps == 0:
                return 0

            assert steps % self.rot_steps_per_round == 0

            if self.rot_rnd_cnt_per_load == 1:
                return 0
            return steps // self.rot_steps_per_round

        def rotation_access_size(self):
            ''' Get total rotation access size. '''
            if self.rot_rnd_cnt_per_load == 1:
                return 0
            return self.rot_access * self.unit_size

        def wide_fetch_access_size(self):
            ''' Get total wide fetch access size. '''
            if self.rot_rnd_cnt_per_load == 1:
                return (self.wf_access + self.saved_wf_access) * self.unit_size
            return self.wf_access * self.unit_size

        def do_access(self, idx_pr, cnt_pr, read=1, write=0):

            ret = self.base.do_access(idx_pr, cnt_pr, read=read, write=write)

            if self.bypass:
                # Bypass, skip buffer sharing.
                return ret

            # Range index.
            ridx_pr = self._range_idx_pr(idx_pr)

            if any(ret):
                # Miss in the shared buffer and load new range. Reset.
                self.cur_rot_unit = 0
                self.rot_step_cnt.setdefault(ridx_pr, 0)

                if self.cur_rot_step_cnt == 0:
                    # Initial fetch, no replaced data yet.
                    assert self.rot_rnd_cnt_per_load is None
                else:
                    rot_rnd_cnt_per_load, rem_ = divmod(
                        self.cur_rot_step_cnt, self.rot_steps_per_round)
                    assert rem_ == 0
                    assert self.rot_rnd_cnt_per_load is None \
                            or self.rot_rnd_cnt_per_load == rot_rnd_cnt_per_load
                    self.rot_rnd_cnt_per_load = rot_rnd_cnt_per_load
                self.cur_rot_step_cnt = 0

            assert all(cnt <= subrng_cnt for cnt, subrng_cnt
                       in zip(cnt_pr, self.subrng_cnt_pr))

            # Subrange index.
            sridx_pr = self._subrange_idx_pr(idx_pr)

            # Rotation unit index.
            ru_idx = self._subrng_rot_unit_idx(sridx_pr)

            if ru_idx != self.cur_rot_unit:
                # Move to next rotation unit.

                if (self.cur_rot_unit + 1) * self.rot_unit_size \
                        >= self.subrng_num:
                    # The current rotation unit is the last one. Start a new
                    # rotation round.
                    # Do not rotate back to the initial state. Instead start
                    # from the current state.
                    self.cur_rot_unit = 0

                    self.last_wf_subrng_idx = 0
                    self.seq_wf_acc = 0

                elif self.cur_rot_unit * self.rot_unit_size \
                        + self.buf_subrng_num >= self.subrng_num:
                    # The last rotation unit is already local. No more rotation.
                    self.cur_rot_unit += 1

                else:
                    # Rotate by one rotation unit, but not exceeding the end.
                    offset = min(self.rot_unit_size,
                                 self.subrng_num
                                 - self.cur_rot_unit * self.rot_unit_size
                                 - self.buf_subrng_num)
                    assert offset > 0

                    # All subranges shift by the above offset.
                    acc_ = (1. * offset / self.buf_subrng_num) * self.subrng_num
                    self.rot_access += util.prod(self.subrng_cnt_pr) * acc_
                    self.cur_rot_unit += 1

                    # One rotation step.
                    self.rot_step_cnt[ridx_pr] += 1
                    self.cur_rot_step_cnt += 1

                    # Combine wide fetch with rotation.
                    self.wf_access -= self.seq_wf_acc
                    self.saved_wf_access += self.seq_wf_acc
                    self.seq_wf_acc = 0

                assert ru_idx == self.cur_rot_unit

            # Buffer index of which has this subrange.
            buf_idx = self._subrng_buf_idx(sridx_pr)

            # Wide fetch from possibly remote buffer.
            wf_acc = util.prod(cnt_pr) * (read + write) * buf_idx
            self.wf_access += wf_acc

            # Record amount of sequential wide fetch.
            subrng_idx = self.subrng_idx_dict[sridx_pr]
            if subrng_idx >= self.last_wf_subrng_idx:
                self.seq_wf_acc += wf_acc
            else:
                self.seq_wf_acc = wf_acc
            self.last_wf_subrng_idx = subrng_idx

            return ret

        def _subrange_idx_pr(self, idx_pr):
            ''' Get the subrange index of all dimensions. '''
            sridx_pr = self.sridx_pr_cache.get(idx_pr, None)
            if sridx_pr is None:
                sridx_pr = tuple((idx % buf_cnt) // subrng_cnt
                                 for idx, buf_cnt, subrng_cnt
                                 in zip(idx_pr, self.buf_cnt_pr,
                                        self.subrng_cnt_pr))
                self.sridx_pr_cache[idx_pr] = sridx_pr
            return sridx_pr

        def _subrng_rot_unit_idx(self, sridx_pr):
            ''' Get the rotation unit index of the subrange. '''
            return self.subrng_idx_dict[sridx_pr] // self.rot_unit_size

        def _subrng_buf_idx(self, sridx_pr):
            ''' Get the buffer index of which currently has the subrange. '''
            subrng_idx = self.subrng_idx_dict[sridx_pr]

            # Start from the current rotation unit.
            subrng_idx -= self.cur_rot_unit * self.rot_unit_size
            subrng_idx %= self.subrng_num

            return self.buf_subrng_centroid[subrng_idx]

        def _init_sub_range(self, lp_t_list, dim_loops):

            assert len(dim_loops) == 2

            subrng_list = [(0, 0)]
            subrng_sz_pr = [1, 1]

            # From inner to outer.
            for lpe, t in reversed(lp_t_list):
                # The data dimension index of this loop.
                try:
                    d = dim_loops.index(lpe)
                except ValueError:
                    # This loop is not related to the data, skip.
                    assert lpe not in dim_loops
                    continue

                # Size of this dimension of current loop body, i.e., all inner
                # loops.
                s = subrng_sz_pr[d]

                # Make the new subrange list, by looping over the current loop
                # body with the current loop factor, and updating this
                # dimension.
                new_subrng_list = []
                for i in range(t):
                    new_subrng_list += [tuple(i_ + i * s if d_ == d else i_
                                              for d_, i_ in enumerate(sr))
                                        for sr in subrng_list]
                subrng_list = new_subrng_list

                # Update size of this dimension.
                subrng_sz_pr[d] *= t

                # Check.
                assert len(set(subrng_list)) == len(subrng_list)
                assert len(subrng_list) == util.prod(subrng_sz_pr)

            subrng_cnt_pr = tuple(buf_cnt // subrng_sz for buf_cnt, subrng_sz
                                  in zip(self.buf_cnt_pr, subrng_sz_pr))

            return subrng_list, subrng_cnt_pr

    def _sim_access_conv(self, lbs, get_bufshr=False):
        '''
        Get data access by actually simulating and generating loops for CONV
        layer.

        If `get_bufshr` is True, also return bufshr stats.
        '''
        self.assertTrue(lbs.is_valid(), '_sim_access_conv: invalid lbs.')

        data_loops = lbs.nld.data_loops

        lpts = tuple(zip(*lbs.bl_ts))

        subgrp_size, rot_unit_cnt, lp_t_list = self._bufshr_params(lbs)
        data_loops = lbs.nld.data_loops

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
            gbufs[dce] = self._SimBufferSharing(
                dce, buf_cnt_pr, lbs.nld.unit_access[me.GBUF][dce],
                subgrp_size[dce], rot_unit_cnt[dce], lp_t_list,
                data_loops[dce].loops(),
                bypass=(not lbs.stored_in_gbuf[dce]))
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

        # Sum over all nodes.
        dram_access = [a * lbs.num_nodes // r for a, r
                       in zip(dram_access, lbs.accfwd_reduction)]
        gbuf_access = [a * lbs.num_nodes for a in gbuf_access]

        # Buffer sharing.
        if get_bufshr:
            rotation_access = [gbufs[dce].rotation_access_size()
                               * (lbs.num_nodes // subgrp_size[dce])
                               for dce in range(de.NUM)]
            wide_fetch_access = [gbufs[dce].wide_fetch_access_size()
                                 * (lbs.num_nodes // subgrp_size[dce])
                                 for dce in range(de.NUM)]
            rotation_rounds = [gbufs[dce].rotation_rounds()
                               for dce in range(de.NUM)]

            return dram_access, gbuf_access, \
                    (rotation_access, wide_fetch_access, rotation_rounds)

        for dce in range(de.NUM):
            self.assertAlmostEqual(gbufs[dce].rotation_access_size(), 0,
                                   msg='_sim_access_conv: non-0 '
                                       'rotation access with no bufshr.')
            self.assertAlmostEqual(gbufs[dce].wide_fetch_access_size(), 0,
                                   msg='_sim_access_conv: non-0 '
                                       'wide fetch access with no bufshr.')
            self.assertEqual(gbufs[dce].rotation_rounds(), 0,
                             msg='_sim_access_conv: non-0 '
                                 'rotation rounds with no bufshr.')

        return dram_access, gbuf_access

    def _average_neighbor_nhops(self, bufshr, subgrp_size):
        ''' Get the average neighbor number of hops. '''

        avg_nbr_nhops = []

        for dce in range(de.NUM):
            # pylint: disable=protected-access

            subgrp_dim, idx_pr = bufshr._subgrp_dim(dce, subgrp_size[dce])
            nbr_dist = bufshr.nbr_dists[dce]

            d_pr = subgrp_dim[idx_pr]
            d_npr = subgrp_dim[1 - idx_pr]
            n_pr = (d_pr - 1) * d_npr
            n_npr = d_npr - 1
            nhops_nbr = bufshr._nhops_with_neighbor_dist(
                dce,
                PhyDim2(*[tpl[1] for tpl
                          in sorted([(idx_pr, n_pr), (1 - idx_pr, n_npr)])]))

            nhops_nbr /= 1. * subgrp_size[dce]

            coord = bufshr._coordinate(subgrp_size[dce] - 1, subgrp_dim, idx_pr)
            nhops_lpbk = bufshr._nhops_with_neighbor_dist(dce, coord)

            nhops_lpbk /= 1. * subgrp_size[dce]

            nhops = nhops_nbr + nhops_lpbk

            if subgrp_size[dce] <= 1:
                self.assertAlmostEqual(nhops, 0)
            elif subgrp_dim.size() == subgrp_size[dce]:
                self.assertTrue(min(nbr_dist) <= nhops
                                <= max(nbr_dist)
                                + 1. * sum(subgrp_dim) / subgrp_dim.size(),
                                '_average_neighbor_nhops: {}: '
                                'subgrp_size {}, subgrp_dim {}, idx_pr {}, '
                                'nbr_dist {}, nhops {} = {} + {}'
                                .format(dce, subgrp_size[dce], subgrp_dim,
                                        idx_pr, nbr_dist,
                                        nhops, nhops_nbr, nhops_lpbk))

            assert not math.isnan(nhops) and not math.isinf(nhops)
            avg_nbr_nhops.append(nhops)

        return avg_nbr_nhops

    def _verify_bufshr_stats(self, dram_access, gbuf_access, bufshr_stats,
                             lbs, bufshr, test_name):
        ''' Verify the buffer sharing stats returned by access simulation. '''

        rotation_access, wide_fetch_access, rotation_rounds = bufshr_stats

        avg_nbr_nhops = self._average_neighbor_nhops(bufshr,
                                                     lbs.bufshr_subgrp_size)

        # Mem hierarchy.
        access = lbs.get_access()

        self.assertListEqual(access[me.DRAM], dram_access,
                             'test_access: DRAM: '
                             'model {} vs. sim {}.'
                             .format(access[me.DRAM], dram_access))
        self.assertListEqual(access[me.GBUF], gbuf_access,
                             'test_access: GBUF: '
                             'model {} vs. sim {}.'
                             .format(access[me.GBUF], gbuf_access))
        self.assertListEqual(access[me.REGF],
                             [lbs.ops, lbs.ops, lbs.ops * 2])

        # NoC.
        noc_access = lbs.get_noc_access()

        for dce in range(de.NUM):
            self.assertAlmostEqual(lbs.bufshr_rotation_access[dce]
                                   + lbs.bufshr_wide_fetch_access[dce],
                                   noc_access[dce])

        for dce in range(de.NUM):
            if lbs.bufshr_subgrp_size[dce] <= 1:
                self.assertAlmostEqual(noc_access[dce], 0)

        for dce in range(de.NUM):
            self.assertAlmostEqual(lbs.bufshr_rot_round_cnt[dce],
                                   rotation_rounds[dce],
                                   msg=('{}: mismatch rotation round count '
                                        'at {}:\nmodel: {}; sim: {}.'
                                        .format(test_name, dce,
                                                lbs.bufshr_rot_round_cnt,
                                                rotation_rounds)))

        for dce in range(de.NUM):
            self.assertAlmostEqual(lbs.bufshr_rotation_access[dce],
                                   rotation_access[dce] * avg_nbr_nhops[dce],
                                   msg=('{}: mismatch NoC rotation access '
                                        'at {}:\nmodel: {}; sim: {} x {}.'
                                        .format(test_name, dce,
                                                lbs.bufshr_rotation_access,
                                                rotation_access,
                                                avg_nbr_nhops)))

        for dce in range(de.NUM):
            self.assertAlmostEqual(lbs.bufshr_wide_fetch_access[dce],
                                   wide_fetch_access[dce] * avg_nbr_nhops[dce],
                                   msg=('{}: mismatch NoC wide fetch access '
                                        'at {}:\nmodel: {}; sim: {} x {}.'
                                        .format(test_name, dce,
                                                lbs.bufshr_wide_fetch_access,
                                                wide_fetch_access,
                                                avg_nbr_nhops)))


    def _regularized_scheme(self, bl_ts, bl_ords):
        ''' Get the regularized scheme which will not be skipped. '''

        assert isinstance(bl_ts, tuple) and isinstance(bl_ords, tuple)
        assert all(isinstance(t, tuple) for t in bl_ts)
        assert all(isinstance(o, tuple) for o in bl_ords)

        reg_lpts = [[] for _ in range(le.NUM)]
        reg_ords = tuple()

        outer_level_innermost_loop = None

        for t_, ord_ in itertools.zip_longest(bl_ts, bl_ords, fillvalue=None):

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

