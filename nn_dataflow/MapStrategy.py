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
import warnings

from . import DataCategoryEnum as de
from . import MemHierEnum as me
from . import Util
from .Layer import Layer, ConvLayer, LocalRegionLayer
from .NestedLoopDesc import NestedLoopDesc
from .PhyDim2 import PhyDim2

class MapStrategy(object):
    '''
    Base mapping strategy.

    Map is the procedure to map the 2D convolution computation onto the 2D PE array.
    '''

    def __init__(self, layer, batch_size, dim_array):
        if not isinstance(layer, Layer):
            raise TypeError('Map: layer must be a Layer object.')
        if not isinstance(dim_array, PhyDim2):
            raise TypeError('Map: dim_array must be a PhyDim2 object.')
        self.layer = layer
        self.batch_size = batch_size
        self.dim_array = dim_array

    def utilization(self):
        '''
        PE utilization, i.e., average percentage of active PEs.
        '''
        raise NotImplementedError('Map: derived class must overwrite.')

    def gen_nested_loop_desc(self):
        '''
        Generate all the NestedLoopDesc objects after mapping.
        '''
        raise NotImplementedError('Map: derived class must overwrite.')


class MapStrategyEyeriss(MapStrategy):
    '''
    Eyeriss mapping scheme, a.k.a, Row-Stationary.

    Chen, Emer, and Sze, ISCA'16.
    '''

    def __init__(self, layer, batch_size, dim_array):

        super(MapStrategyEyeriss, self).__init__(layer, batch_size, dim_array)

        # Logic PE set.
        if isinstance(self.layer, ConvLayer):
            # Conv and FC layers.
            self.ops_lpe = self.layer.sfil * self.layer.wofm
            self.dim_lpeset = PhyDim2(self.layer.sfil, self.layer.hofm)
            cnt_lpeset = self.batch_size * self.layer.nofm * self.layer.nifm
        elif isinstance(self.layer, LocalRegionLayer):
            self.ops_lpe = self.layer.ops_per_neuron()
            self.dim_lpeset = PhyDim2(h=self.layer.hofm, w=self.layer.wofm)
            cnt_lpeset = self.batch_size * self.layer.nofm
        else:
            raise TypeError('MapEyeriss: unrecognized layer type {}.'
                            .format(type(self.layer)))

        ops_logic_total = self.ops_lpe * self.dim_lpeset.size() * cnt_lpeset
        assert ops_logic_total == self.layer.total_ops(self.batch_size)

        # Physical PE set through replication and folding.
        self._repl_fold()

        # PE utilization.
        # We replicate repl.size() lpesets, and then fold to fold.size()
        # physical array passes.
        self.util = 1. * (self.dim_lpeset.size() * self.repl.size()) \
                / (self.dim_array.size() * self.fold.size())
        assert self.util <= 1. + 1e-6

        if self.util < 0.5:
            warnings.warn('MapEyeriss: PE array resource utilization < 50%. '
                          'Physical PE set {}; array size {}; logic PE set {}; '
                          'folded logic PE set {}. Can\'t we fit more?'
                          .format(self.dim_ppeset, self.dim_array,
                                  self.dim_lpeset, self.dim_flpeset))

    def utilization(self):
        return self.util

    def gen_nested_loop_desc(self):
        # NOTE: not sure if it is worth to merge the two functions, as there is
        # significant redundancy but also significant difference.
        if isinstance(self.layer, ConvLayer):
            for nld in self._gen_nested_loop_desc_conv():
                yield nld
        elif isinstance(self.layer, LocalRegionLayer):
            for nld in self._gen_nested_loop_desc_localregion():
                yield nld

    def _gen_nested_loop_desc_conv(self):
        # repl.w can only be used for ofm; repl.h can be shared by ifm and ofm.
        #
        # fold.h folds fil, which uses different parts of fil but same ifm and
        # ofm, so do these ppeset continuously (innermost loop); fold.w folds
        # ofm (and ifm), which uses different parts of ofm and ifm but same
        # fil, so merge into batch size.

        # Terminologies:
        #
        # flpeset: folded lpeset, a fraction of lpeset after folding and before
        # replication.
        #
        # ppeset: physical peset, one physical array pass, replicated flpeset.
        #
        # procpass: processing pass, fold.h number of ppesets, i.e., fold.h
        # physical array passes, which deals with all folded fils, one folded
        # ifm and one folded ofm. A procpass includes all replication. See
        # Chen, et al, ISCA'16, end of V.B.
        #
        # # ppesets in a procpass = fold.h.
        # # flpesets in a procpass = fold.h * repl.size().
        ppesets_per_procpass = self.fold.h
        # To iterate all folded fils over the fmaps, we have two choices:
        # a) only store one folded fil in regf and access fmaps multiple times
        # from gbuf;
        # b) store all folded fils in regf and only access fmaps once from
        # gbuf.
        # To save regf size, we choose a).
        #
        # Access rounds: the times each single data element is accessed.
        accrnds_per_procpass = [float('nan')] * de.NUM
        accrnds_per_procpass[de.FIL] = 1
        accrnds_per_procpass[de.IFM] = ppesets_per_procpass
        accrnds_per_procpass[de.OFM] = ppesets_per_procpass

        # Processing pass is the unit for loop blocking, i.e., the innermost
        # loop processes one procpass. So the unit accesses are also calculated
        # on procpass.

        # Average (considering occupation) number of PEs for one ppeset.
        avgpes_per_ppeset = self.dim_array.size() * self.util

        # Average (considering occupation) number of ops for one procpass.
        avgops_per_procpass = ppesets_per_procpass * avgpes_per_ppeset \
                * self.ops_lpe

        # Average (considering occupation) number of rows for one flpeset.
        # Row size is not affected by folding/replication since a row is within
        # one PE.
        avgrows_per_flpeset = [float('nan')] * de.NUM
        # Reduced by folding factor.
        avgrows_per_flpeset[de.FIL] = 1. * self.dim_lpeset.h / self.fold.h
        # Reduced by folding factor.
        avgrows_per_flpeset[de.OFM] = 1. * self.dim_lpeset.w / self.fold.w
        # Determined by fil and ofm rows.
        avgrows_per_flpeset[de.IFM] = avgrows_per_flpeset[de.FIL] \
                + (avgrows_per_flpeset[de.OFM] - 1) * self.layer.htrd
        # For ifmap with strides, there may be gaps in ifmap which are not
        # needed in ofmap. In such case, the actual needed # ifmap rows is
        # bound by # ofmap rows * # filter rows, since each ofmap row needs
        # ifmap rows equal to the filter rows.
        avgrows_per_flpeset[de.IFM] = min(avgrows_per_flpeset[de.IFM],
                                          avgrows_per_flpeset[de.OFM]
                                          * avgrows_per_flpeset[de.FIL])
        assert avgrows_per_flpeset[de.FIL] <= self.dim_flpeset.h + 1e-6
        assert avgrows_per_flpeset[de.OFM] <= self.dim_flpeset.w + 1e-6
        # Due to folding, the overlapping ifmaps may need to be re-fetched,
        # resulting in amplified access for ifmaps. On the other hand, if
        # the stride results in gaps in ifmaps, some ifmaps are not accessed.
        # Consider one flpeset, hifm rows are folded by fold.size(), but each
        # is accessed accrnds_per_procpass times.
        amp_acc_ifm = 1. * avgrows_per_flpeset[de.IFM] * self.fold.size() \
                / (self.layer.hifm * accrnds_per_procpass[de.IFM])

        # Unit regf size for one processing pass.
        usz_regf = [0] * de.NUM
        # Entire fil row per PE, and only store one folded fil because we
        # access fmaps multiple times.
        usz_regf[de.FIL] = self.layer.sfil
        # For 1D conv in each PE, ifm and ofm are both accessed in a streaming
        # fashion (sliding window). Only capture sfil ifm elements and 1 ofm
        # element is adequate.
        usz_regf[de.IFM] = self.layer.sfil
        usz_regf[de.OFM] = 1
        usize_regf = tuple(usz_regf)

        # The total size of accessed data across all PE regfs for one
        # processing pass, including duplication.
        avgsize_all_regfs = [avgpes_per_ppeset * ar
                             for ar in accrnds_per_procpass]
        # Entire fil row per PE, also store all folded fils.
        avgsize_all_regfs[de.FIL] *= self.layer.sfil * ppesets_per_procpass
        # Entire ifm row per PE.
        avgsize_all_regfs[de.IFM] *= self.layer.wifm
        # Entire ofm row per PE.
        avgsize_all_regfs[de.OFM] *= self.layer.wofm

        # Time.
        unit_time = ppesets_per_procpass * self.ops_lpe

        # Loop body unit time is constant w.r.t. the split of repl.h, so we
        # pick the smallest total number of loops.
        min_cnt_loops = float('inf')

        for t_repl_h in Util.factorize(self.repl.h, 2):

            # Determine the numbers of i/ofmaps per processing pass.
            # repl.w is only used for ofmaps, and repl.h can be used either for
            # ifmaps or ofmaps.
            ifms_per_procpass = t_repl_h[0]
            ofms_per_procpass = t_repl_h[1] * self.repl.w

            # Loop trip counts.
            # fold.w is equivalent to increasing batch size (and fold.h is
            # within processing pass).
            lcnt_ifm = Util.idivc(self.layer.nifm, ifms_per_procpass)
            lcnt_ofm = Util.idivc(self.layer.nofm, ofms_per_procpass)
            lcnt_bat = self.batch_size * self.fold.w

            cnt_loops = lcnt_ifm * lcnt_ofm * lcnt_bat
            if cnt_loops < min_cnt_loops:
                min_cnt_loops = cnt_loops
            elif cnt_loops > min_cnt_loops:
                continue

            # Loop occupation.
            # This is due to partial full loops. E.g., for total of 32 ifmaps,
            # if each loop body processes 3, we need 10 loops, but the last one
            # only has 2/3 ifmaps.
            locc_ifm = 1. * self.layer.nifm / ifms_per_procpass / lcnt_ifm
            locc_ofm = 1. * self.layer.nofm / ofms_per_procpass / lcnt_ofm
            locc_bat = 1. * self.batch_size * self.fold.w / lcnt_bat

            # Unit gbuf size for one processing pass. Ceil the number of rows.
            usize_gbuf = self._calc_unit_size_gbuf_conv(
                [int(math.ceil(r) + 1e-6) for r in avgrows_per_flpeset],
                ifms_per_procpass, ofms_per_procpass, ppesets_per_procpass)

            # Loop occupations affect accesses.
            # Total accesses = avg unit accesses * loop count.
            # Avg unit accesses = full-loop unit accesses * occupation.
            #
            # Loop occupations do not affect size, since size needs to be the
            # maximum, i.e., full loop case.
            occ_acc = [0] * de.NUM
            occ_acc[de.FIL] = locc_ifm * locc_ofm
            occ_acc[de.IFM] = locc_ifm * locc_bat
            occ_acc[de.OFM] = locc_ofm * locc_bat

            # Unit access, i.e., number of data element accesses for one
            # processing pass. This is the average over all loops, considering
            # loop occupations.
            uaccess = [tuple() for _ in range(me.NUM)]
            # DRAM access is based on gbuf, need to buffer all data in gbuf.
            uacc_gbuf = self._calc_unit_size_gbuf_conv(avgrows_per_flpeset,
                                                       ifms_per_procpass,
                                                       ofms_per_procpass,
                                                       ppesets_per_procpass)
            uaccess[me.DRAM] = tuple(ua * o for ua, o
                                     in zip(uacc_gbuf, occ_acc))
            # gbuf access.
            # Load each element once from gbuf then use itcn.
            uaccess[me.GBUF] = tuple(ua * ar for ua, ar
                                     in zip(uaccess[me.DRAM],
                                            accrnds_per_procpass))
            # itcn access is total accessed regf size - gbuf access.
            uaccess[me.ITCN] = tuple(asar - ua for asar, ua
                                     in zip(avgsize_all_regfs,
                                            uaccess[me.GBUF]))
            assert all(ua >= 0 for ua in uaccess[me.ITCN]), \
                    'MapEyeriss: encounter negative access count to itcn {}' \
                    .format(uaccess[me.ITCN])
            # regf access is based on num ops.
            uaccess[me.REGF] = (avgops_per_procpass,) * de.NUM

            # Check unit access.
            Util.assert_float_eq_int(
                uaccess[me.DRAM][de.FIL] * lcnt_ifm * lcnt_ofm,
                self.layer.total_filter_size(),
                'MapEyeriss: unit access at DRAM for FIL {} is incorrect.'
                .format(uaccess[me.DRAM][de.FIL]))
            Util.assert_float_eq_int(
                # Need to consider amplified access for IFM.
                uaccess[me.DRAM][de.IFM] * lcnt_ifm * lcnt_bat / amp_acc_ifm,
                self.layer.total_ifmap_size() * self.batch_size,
                'MapEyeriss: unit access at DRAM for IFM {} is incorrect.'
                .format(uaccess[me.DRAM][de.IFM]))
            Util.assert_float_eq_int(
                uaccess[me.DRAM][de.OFM] * lcnt_ofm * lcnt_bat,
                self.layer.total_ofmap_size() * self.batch_size,
                'MapEyeriss: unit access at DRAM for OFM {} is incorrect.'
                .format(uaccess[me.DRAM][de.OFM]))

            # Finalize unit access.
            unit_access = tuple(uaccess)

            # Num of ops. In addition to procpass utilization, also add the
            # impact of loop occupations.
            unit_ops = avgops_per_procpass * locc_ifm * locc_ofm * locc_bat

            # Check num of ops.
            ops_physical_total = unit_ops * lcnt_ifm * lcnt_ofm * lcnt_bat

            Util.assert_float_eq_int(
                ops_physical_total, self.layer.total_ops(self.batch_size),
                'MapEyeriss: total number of physical ops is incorrect.')

            yield NestedLoopDesc(loopcnt_ifm=lcnt_ifm, loopcnt_ofm=lcnt_ofm,
                                 loopcnt_bat=lcnt_bat, usize_gbuf=usize_gbuf,
                                 usize_regf=usize_regf, unit_access=unit_access,
                                 unit_ops=unit_ops, unit_time=unit_time)

    def _gen_nested_loop_desc_localregion(self):
        # Terminologies:
        #
        # flpeset: folded lpeset, a fraction of lpeset after folding and before
        # replication.
        #
        # ppeset: physical peset, one physical array pass.
        #
        # procpass: processing pass, one ppeset, which deals with one folded
        # ofm. A procpass includes all replication.
        ppesets_per_procpass = 1

        # Processing pass is the unit for loop blocking, i.e., the innermost
        # loop processes one procpass. So the unit accesses are also calculated
        # on procpass.

        # Average (considering occupation) number of PEs for one ppeset.
        avgpes_per_ppeset = self.dim_array.size() * self.util

        # Average (considering occupation) number of ops for one procpass.
        avgops_per_procpass = ppesets_per_procpass * avgpes_per_ppeset \
                * self.ops_lpe

        # Average (considering occupation) dims of i/ofmap for one flpeset.
        avgdims_per_flpeset = [(float('nan'), float('nan'))
                               for _ in range(de.NUM)]
        # No sfil.
        avgdims_per_flpeset[de.FIL] = (0, 0)
        # Reduced by folding factor.
        avgdims_per_flpeset[de.OFM] = (1. * self.dim_lpeset.h / self.fold.h,
                                       1. * self.dim_lpeset.w / self.fold.w)
        # Determined by ofm dims and region.
        avgdims_per_flpeset[de.IFM] = (
            self.layer.hreg + (avgdims_per_flpeset[de.OFM][0] - 1) \
                    * self.layer.htrd,
            self.layer.wreg + (avgdims_per_flpeset[de.OFM][1] - 1) \
                    * self.layer.wtrd)
        # Due to folding, the overlapping ifmaps may need to be re-fetched,
        # resulting in amplified access for ifmaps.
        amp_acc_ifm = 1. * Util.prod(avgdims_per_flpeset[de.IFM]) \
                * self.fold.size() / self.layer.ifmap_size()

        # Unit regf size for one processing pass.
        usz_regf = [0] * de.NUM
        # No sfil.
        usz_regf[de.FIL] = 0
        # Similar to a line buffer, ifm needs to store the (sliding) region,
        # while ofm can have only 1 element.
        usz_regf[de.IFM] = self.layer.region_size()
        usz_regf[de.OFM] = 1
        usize_regf = tuple(usz_regf)

        # The total size of accessed data across all PE regfs for one
        # processing pass, including duplication.
        avgsize_all_regfs = [avgpes_per_ppeset] * de.NUM
        # No sfil.
        avgsize_all_regfs[de.FIL] *= 0
        # Since ifm loop count is always 1, we account for all ifmap accesses.
        # This contains all fmaps, and the regions in each fmap.
        avgsize_all_regfs[de.IFM] *= self.layer.nifm \
                * self.layer.hreg * self.layer.wreg
        # Entire ofm.
        avgsize_all_regfs[de.OFM] *= 1

        # Time.
        unit_time = ppesets_per_procpass * self.ops_lpe


        # We don't have ifm loop, only ofm and bat loops.
        # We use all repl for ofm loop to exploit ifm reuse on regf level.
        # Since there is no fil and no fil reuse in batch, repl for bat loop
        # has no benefits.
        ofms_per_procpass = self.repl.size()

        # Loop trip counts.
        # ifm loop count is always 1, i.e., only exists ofm loop.
        # fold is equivalent to increasing batch size.
        lcnt_ifm = 1
        lcnt_ofm = Util.idivc(self.layer.nofm, ofms_per_procpass)
        lcnt_bat = self.batch_size * self.fold.size()

        # Loop occupation.
        # This is due to partial full loops. E.g., for total of 32 ifmaps,
        # if each loop body processes 3, we need 10 loops, but the last one
        # only has 2/3 ifmaps.
        locc_ifm = 1.
        locc_ofm = 1. * self.layer.nofm / ofms_per_procpass / lcnt_ofm
        locc_bat = 1. * self.batch_size * self.fold.size() / lcnt_bat

        # Unit gbuf size for one processing pass. Ceil the number of rows.
        usize_gbuf = self._calc_unit_size_gbuf_localregion(
            avgdims_per_flpeset, ofms_per_procpass)

        # Loop occupations affect accesses.
        # Total accesses = avg unit accesses * loop count.
        # Avg unit accesses = full-loop unit accesses * occupation.
        #
        # Loop occupations do not affect size, since size needs to be the
        # maximum, i.e., full loop case.
        occ_acc = [0] * de.NUM
        occ_acc[de.FIL] = locc_ifm * locc_ofm
        occ_acc[de.IFM] = locc_ifm * locc_bat
        occ_acc[de.OFM] = locc_ofm * locc_bat

        # Unit access, i.e., number of data element accesses for one
        # processing pass. This is the average over all loops, considering
        # loop occupations.
        uaccess = [tuple() for _ in range(me.NUM)]
        # DRAM access is based on gbuf, need to buffer all data in gbuf.
        uacc_gbuf = self._calc_unit_size_gbuf_localregion(
            avgdims_per_flpeset, ofms_per_procpass, is_access=True)
        uaccess[me.DRAM] = tuple(ua * o for ua, o
                                 in zip(uacc_gbuf, occ_acc))
        # gbuf access.
        # Load each element once from gbuf then use itcn.
        uaccess[me.GBUF] = tuple(ua for ua in uaccess[me.DRAM])
        # itcn access is total accessed regf size - gbuf access.
        uaccess[me.ITCN] = tuple(asar - ua for asar, ua
                                 in zip(avgsize_all_regfs, uaccess[me.GBUF]))
        assert all(ua >= 0 for ua in uaccess[me.ITCN]), \
                'MapEyeriss: encounter negative access count to itcn {}' \
                .format(uaccess[me.ITCN])
        # regf access is based on num ops.
        uaccess[me.REGF] = (avgops_per_procpass,) * de.NUM

        # Check unit access.
        Util.assert_float_eq_int(
            uaccess[me.DRAM][de.FIL] * lcnt_ifm * lcnt_ofm,
            0,
            'MapEyeriss: unit access at DRAM for FIL {} is incorrect.'
            .format(uaccess[me.DRAM][de.FIL]))
        Util.assert_float_eq_int(
            # Need to consider amplified access for IFM.
            uaccess[me.DRAM][de.IFM] * lcnt_ifm * lcnt_bat / amp_acc_ifm,
            self.layer.total_ifmap_size() * self.batch_size,
            'MapEyeriss: unit access at DRAM for IFM {} is incorrect.'
            .format(uaccess[me.DRAM][de.IFM]))
        Util.assert_float_eq_int(
            uaccess[me.DRAM][de.OFM] * lcnt_ofm * lcnt_bat,
            self.layer.total_ofmap_size() * self.batch_size,
            'MapEyeriss: unit access at DRAM for OFM {} is incorrect.'
            .format(uaccess[me.DRAM][de.OFM]))

        # Finalize unit access.
        unit_access = tuple(uaccess)


        # Num of ops. In addition to procpass utilization, also add the
        # impact of loop occupations.
        unit_ops = avgops_per_procpass * locc_ifm * locc_ofm * locc_bat

        # Check num of ops.
        ops_physical_total = unit_ops * lcnt_ifm * lcnt_ofm * lcnt_bat

        Util.assert_float_eq_int(
            ops_physical_total, self.layer.total_ops(self.batch_size),
            'MapEyeriss: total number of physical ops is incorrect.')

        yield NestedLoopDesc(loopcnt_ifm=lcnt_ifm, loopcnt_ofm=lcnt_ofm,
                             loopcnt_bat=lcnt_bat, usize_gbuf=usize_gbuf,
                             usize_regf=usize_regf, unit_access=unit_access,
                             unit_ops=unit_ops, unit_time=unit_time)

    def _repl_fold(self):
        '''
        Find the replication and folding factors from logic PE set to physical
        array.
        '''
        fold_w = 1
        repl_w = 1
        fold_h = 1
        repl_h = 1

        if self.dim_lpeset.h > self.dim_array.h:
            # Fold on height.
            fold_h = Util.idivc(self.dim_lpeset.h, self.dim_array.h)
        else:
            # Replicate on height.
            repl_h = self.dim_array.h // self.dim_lpeset.h
        if self.dim_lpeset.w > self.dim_array.w:
            # Fold on width.
            fold_w = Util.idivc(self.dim_lpeset.w, self.dim_array.w)
        else:
            # Replicate on with.
            repl_w = self.dim_array.w // self.dim_lpeset.w

        # Adjust fold and repl, use repl_h to first schedule fold_w.
        # The factor of putting fold_w to repl_h (w to h) is the smaller of the
        # two. Either repl_h cannot accommodate all fold_w, still fold_w; or
        # repl_h has accommodated all fold_w, remain repl_h.
        f_w2h = min(repl_h, fold_w)
        fold_w = Util.idivc(fold_w, f_w2h)
        repl_h = repl_h // f_w2h

        # The replication and folding factors for lpeset, considering the
        # adjustment.
        self.fold = PhyDim2(fold_h, fold_w)
        self.repl = PhyDim2(repl_h, repl_w)

        # The folded lpeset size on the ppeset after adjustment. The width may
        # be larger than the array width, but it is actually broken into the
        # height replication.
        self.dim_flpeset = PhyDim2(Util.idivc(self.dim_lpeset.h, self.fold.h),
                                   Util.idivc(self.dim_lpeset.w, self.fold.w))

        # The physical ppeset size, should fit in the array.
        self.dim_ppeset = PhyDim2(self.dim_flpeset.h * self.repl.h * f_w2h,
                                  Util.idivc(self.dim_flpeset.w * self.repl.w,
                                             f_w2h))

        if not (self.dim_ppeset.h <= self.dim_array.h
                and self.dim_ppeset.w <= self.dim_array.w):
            raise RuntimeError('MapEyeriss: dim_ppeset with size {} does not '
                               'fit in dim_array with size {}.'
                               .format(self.dim_ppeset, self.dim_array))

    def _calc_unit_size_gbuf_conv(self, rows_per_flpeset, ifms_per_procpass,
                                  ofms_per_procpass, ppesets_per_procpass):
        '''
        Calculate the unit gbuf size for one processing pass.
        '''
        usize_gbuf = [0] * de.NUM
        # Size = row size * # rows per flpeset * flpeset count in procpass.
        # Only fil needs to multiply ppesets_per_procpass, because for
        # each ppeset in the processing pass, fil is different (folded
        # parts) while ifm and ofm are the same.
        usize_gbuf[de.FIL] = (self.layer.sfil
                              * rows_per_flpeset[de.FIL]
                              * (ifms_per_procpass
                                 * ofms_per_procpass
                                 * ppesets_per_procpass))
        usize_gbuf[de.IFM] = (self.layer.wifm
                              * rows_per_flpeset[de.IFM]
                              * ifms_per_procpass)
        usize_gbuf[de.OFM] = (self.layer.wofm
                              * rows_per_flpeset[de.OFM]
                              * ofms_per_procpass)
        return tuple(usize_gbuf)

    def _calc_unit_size_gbuf_localregion(self, avgdims_per_flpeset,
                                         ofms_per_procpass, is_access=False):
        '''
        Calculate the unit gbuf size for one processing pass.
        '''
        if is_access:
            dims_per_flpeset = avgdims_per_flpeset
        else:
            # Ceil the number of rows.
            dims_per_flpeset = [tuple(int(math.ceil(d) + 1e-6) for d in dims)
                                for dims in avgdims_per_flpeset]

        # Number of ifmap needed.
        if is_access:
            # Since ifm loop count is always 1, we account for all ifmaps
            # accesses here.
            ifms_per_procpass = self.layer.nifm
        else:
            # Determined by nreg.
            ifms_per_procpass = ofms_per_procpass - 1 + self.layer.nreg

        usize_gbuf = [0] * de.NUM
        # No sfil.
        usize_gbuf[de.FIL] = 0
        usize_gbuf[de.IFM] = (Util.prod(dims_per_flpeset[de.IFM])
                              * ifms_per_procpass)
        usize_gbuf[de.OFM] = (Util.prod(dims_per_flpeset[de.OFM])
                              * ofms_per_procpass)
        return tuple(usize_gbuf)

