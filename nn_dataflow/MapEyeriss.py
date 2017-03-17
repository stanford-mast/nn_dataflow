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
Eyeriss mapping scheme, a.k.a, Row-Stationary.

Chen, Emer, and Sze, ISCA'16.
'''

import warnings
from operator import sub, mul

from . import DataCategoryEnum as de
from . import LoopBlocking
from . import MemHierEnum as me
from . import Util
from .PhyDim2 import PhyDim2

def gen_nested_loop_desc(layer, batch_size, dim_array):
    '''
    Using Eyeriss mapping to formalize into a loop blocking problem.
    '''

    # Logic PE set.
    dim_lpeset = PhyDim2(layer.sfil, layer.hofm)
    cnt_lpeset = batch_size * layer.nofm * layer.nifm

    ops_lpe = layer.sfil * layer.wofm
    ops_lpeset = ops_lpe * dim_lpeset.size()
    ops_logic_total = ops_lpeset * cnt_lpeset

    def repl_fold(layer, dim_lpeset, dim_array):
        '''
        Return replication and folding factors from logic PE set to physical
        array.
        '''
        fold_w = 1
        repl_w = 1
        fold_h = 1
        repl_h = 1
        # Basic fold and repl.
        if dim_lpeset.h > dim_array.h:
            # Fold on height.
            fold_h = Util.idivc(dim_lpeset.h, dim_array.h)
        else:
            # Replicate on height.
            repl_h = dim_array.h // dim_lpeset.h
            # Try make repl_h divide nifm * nofm because of tiling.
            x = repl_h
            while x > repl_h / 2:
                if Util.approx_dividable(layer.nifm * layer.nofm, x, 0.1):
                    repl_h = x
                    break
                x -= 1
        if dim_lpeset.w > dim_array.w:
            # Fold on width.
            fold_w = Util.idivc(dim_lpeset.w, dim_array.w)
        else:
            # Replicate on with.
            repl_w = dim_array.w // dim_lpeset.w
            # Try make repl_w divide nofm because of tiling.
            x = repl_w
            while x > repl_w / 2:
                if Util.approx_dividable(layer.nofm, x, 0.1):
                    repl_w = x
                    break
                x -= 1
        # Adjust fold and repl, use repl_h to first schedule fold_w.
        # The factor of putting fold_w to repl_h (w to h) is the smaller of the
        # two. Either repl_h cannot accommodate all fold_w, still fold_w; or
        # repl_h has accommodated all fold_w, remain repl_h.
        f_w2h = min(repl_h, fold_w)
        fold_w = Util.idivc(fold_w, f_w2h)
        repl_h = repl_h // f_w2h

        fold = PhyDim2(fold_h, fold_w)
        repl = PhyDim2(repl_h, repl_w)

        # dim_ppeset_frac is the physical PE set fraction dimensions.
        # Fraction is the unit before replication. The h dim is # fil rows, and
        # the w dim is # ofm rows.
        dim_ppeset_frac = PhyDim2(Util.idivc(dim_lpeset.h, fold_h),
                                  Util.idivc(dim_lpeset.w, fold_w))
        # Physical PE set dimensions.
        dim_ppeset = PhyDim2(dim_ppeset_frac.h * repl_h * f_w2h,
                             Util.idivc(dim_ppeset_frac.w * repl_w, f_w2h))

        return dim_ppeset, repl, fold, dim_ppeset_frac


    dim_ppeset, repl, fold, dim_ppeset_frac \
            = repl_fold(layer, dim_lpeset, dim_array)

    if not (dim_ppeset.h <= dim_array.h and dim_ppeset.w <= dim_array.w):
        raise RuntimeError('MapEyeriss: dim_ppeset with size {} does not fit '
                           'in dim_array with size {}'
                           .format(dim_ppeset, dim_array))
    # By now we have fold, repl and dim_ppeset.
    # fold.size() * dim_ppeset.size() >~ repl.size() * dim_lpeset.size()

    resource_utilization = dim_ppeset.size() / float(dim_array.size())
    if resource_utilization < 0.25:
        raise RuntimeError('MapEyeriss: array resource utilization is < 25%. '
                           'Physical PE set {}; array size {}. '
                           'Can\'t we fit more?'.format(dim_ppeset, dim_array))
    elif resource_utilization < 0.5:
        warnings.warn('MapEyeriss: array resource utilization is < 50%.')

    # repl.w can only be used for ofm; repl.h can be shared by ifm and ofm.
    # fold.h folds fil, which uses different parts of fil but same ifm and ofm,
    # so do these ppeset continuously (innermost loop); fold.w folds ofm (and
    # ifm), which uses different parts of ofm and ifm but same fil, so do
    # separately (outermost loop).

    # Define a processing pass (see Chen, et al, ISCA'16, end of V.B) as all
    # fold.h ppesets, with all folded fil, one folded ifm and one folded ofm.
    # So in one processing pass, # rows of fil is the same as original (folded
    # but all folded fil are included), # rows of ifm/ofm is reduced to 1 over
    # fold factor (1 / fold.h). However, row size is not changed (not affected
    # by folding/replication as within one PE).
    # Processing pass is the unit for loop blocking, i.e., the innermost loop
    # process one processing pass.
    cnt_ppesets_per_procpass = fold.h

    ops_procpass = cnt_ppesets_per_procpass * dim_ppeset.size() * ops_lpe

    ## Unit regf size for one processing pass.

    usize_regf = [dim_array.size()] * de.NUM
    # Entire fil row per PE, also store all folded fil.
    usize_regf[de.FIL] *= layer.sfil * cnt_ppesets_per_procpass
    # Per each PE, for 1D conv, ifm and ofm are both accessed with a streaming
    # fashion (sliding window).  Only capture sfil ifm elements and 1 ofm
    # element is adequate.
    usize_regf[de.IFM] *= layer.sfil
    usize_regf[de.OFM] *= 1

    # Also determine the total access data size from regf.
    usize_total_regf = [dim_ppeset.size()] * de.NUM
    # Entire fil row per PE, also store all folded fil.
    usize_total_regf[de.FIL] *= layer.sfil * cnt_ppesets_per_procpass
    # Entire ifm row per PE.
    usize_total_regf[de.IFM] *= layer.wifm
    # Entire ofm row per PE.
    usize_total_regf[de.OFM] *= layer.wofm

    # Number of rows needed in frac ppeset.
    num_rows_ppeset_frac = [0] * de.NUM
    num_rows_ppeset_frac[de.FIL] = dim_ppeset_frac.h
    num_rows_ppeset_frac[de.OFM] = dim_ppeset_frac.w
    num_rows_ppeset_frac[de.IFM] = (num_rows_ppeset_frac[de.FIL]
                                    + (num_rows_ppeset_frac[de.OFM] - 1)
                                    * layer.strd)

    # For ifmap with strides, there may be gaps in ifmap which are not needed
    # in ofmap. In such case, # PEs in frac ppeset, i.e.,
    # dim_ppeset_frac.size() will be smaller than num_rows_ppeset_frac[de.IFM].
    # The actual needed # rows in frac ppeset is bound by
    # dim_ppeset_frac.size().
    num_rows_ppeset_frac[de.IFM] = min(num_rows_ppeset_frac[de.IFM],
                                       dim_ppeset_frac.size())

    for t_repl_h in Util.factorize(repl.h, 2):

        cnt_ifms_per_ppeset = t_repl_h[0]
        cnt_ofms_per_ppeset = t_repl_h[1] * repl.w

        ## Loop trip counts.

        if cnt_ifms_per_ppeset > 1.2 * layer.nifm:
            continue
        if cnt_ofms_per_ppeset > 1.2 * layer.nofm:
            continue

        lcnt_ifm = Util.idivc(layer.nifm, cnt_ifms_per_ppeset)
        lcnt_ofm = Util.idivc(layer.nofm, cnt_ofms_per_ppeset)
        lcnt_bat = batch_size * fold.w

        ## Unit gbuf size for one processing pass.

        usize_gbuf = [0] * de.NUM
        # Size = # rows * row size * count in procpass.
        # Only fil needs to multiply cnt_ppesets_per_procpass, because for each
        # ppeset in the processing pass, fil is different while ifm and ofm are
        # the same.
        usize_gbuf[de.FIL] = num_rows_ppeset_frac[de.FIL] \
                            * layer.sfil \
                            * (cnt_ifms_per_ppeset * cnt_ofms_per_ppeset
                               * cnt_ppesets_per_procpass)
        usize_gbuf[de.IFM] = num_rows_ppeset_frac[de.IFM] \
                            * layer.wifm \
                            * cnt_ifms_per_ppeset
        usize_gbuf[de.OFM] = num_rows_ppeset_frac[de.OFM] \
                            * layer.wofm \
                            * cnt_ofms_per_ppeset

        ## Number of data element accesses for physical PE set.

        unit_access = [0] * me.NUM
        # DRAM access is based on gbuf, need to buffer all data in gbuf.
        unit_access[me.DRAM] = map(mul, usize_gbuf, [1, 1, 2])
        # gbuf access.
        # Load each element once from gbuf then use itcn.
        # ofm read and write once each.
        unit_access[me.GBUF] = map(mul, usize_gbuf, [1, 1, 2])
        # itcn access is total transfer size - gbuf access.
        # Transfer size is per-PE regf size * ppeset size.
        transz_procpass = map(mul, usize_total_regf, [1, 1, 2])
        unit_access[me.ITCN] = map(sub, transz_procpass, unit_access[me.GBUF])
        if not all([x >= 0 for x in unit_access[me.ITCN]]):
            raise RuntimeError('MapEyeriss: encounter negative access count to '
                               'itcn {}'.format(unit_access[me.ITCN]))
        # regf access is based on num ops.
        unit_access[me.REGF] = map(mul, [ops_procpass] * 3, [1, 1, 2])

        ## Num of ops.

        unit_ops = ops_procpass
        ops_physical_total = unit_ops * lcnt_ifm * lcnt_ofm * lcnt_bat
        # ops_logic_total ~< ops_physical_total
        if ops_physical_total / float(ops_logic_total) > 2:
            raise RuntimeError('MapEyeriss: # physical ops ({}) is more than '
                               'double of # logic ops ({})'
                               .format(ops_physical_total, ops_logic_total))

        unit_time = cnt_ppesets_per_procpass * ops_lpe

        yield LoopBlocking.NestedLoopDesc(lcnt_ifm, lcnt_ofm, lcnt_bat,
                                          usize_gbuf, usize_regf,
                                          unit_access, unit_ops, unit_time)

