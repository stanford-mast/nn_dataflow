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

from . import DataCategoryEnum as de
from . import ParallelEnum as pe
from .PhyDim2 import PhyDim2
from .Util import StringifyClass

class BufShrScheme(StringifyClass):
    '''
    The buffer sharing scheme.
    '''

    def __init__(self, part):

        # Dimension of the node group.
        self.dims = [PhyDim2(float('nan'), float('nan'))] * de.NUM
        # Distance between the neighbors in the node group.
        self.nbr_dists = [PhyDim2(float('nan'), float('nan'))] * de.NUM

        # Fil weights in gbuf can be shared by nodes with OFMP and BATP.
        idx_ofmp = part.order.index(pe.OFMP)
        idx_batp = part.order.index(pe.BATP)
        dim_ofmp = part.dim(pe.OFMP)
        dim_batp = part.dim(pe.BATP)
        # If only one of OFMP and BATP exists, use that one.
        if dim_ofmp.size() == 1:
            self.dims[de.FIL] = dim_batp
            self.nbr_dists[de.FIL] = part.part_neighbor_dist(pe.BATP)
        elif dim_batp.size() == 1:
            self.dims[de.FIL] = dim_ofmp
            self.nbr_dists[de.FIL] = part.part_neighbor_dist(pe.OFMP)
        else:
            # If both exist ...
            if abs(idx_ofmp - idx_batp) == 1:
                # ... and are adjacent in the partitioning hierarchy, use
                # both.
                self.dims[de.FIL] = dim_batp * dim_ofmp
                self.nbr_dists[de.FIL] = part.part_neighbor_dist(
                    pe.OFMP if idx_ofmp > idx_batp else pe.BATP)
            else:
                # ... but are not adjacent, use the bottom one (with
                # smaller distance).
                if idx_ofmp > idx_batp:
                    self.dims[de.FIL] = dim_ofmp
                    self.nbr_dists[de.FIL] = part.part_neighbor_dist(pe.OFMP)
                else:
                    self.dims[de.FIL] = dim_batp
                    self.nbr_dists[de.FIL] = part.part_neighbor_dist(pe.BATP)

        # Ifmaps in gbuf can be shared by nodes with OUTP.
        self.dims[de.IFM] = part.dim(pe.OUTP)
        self.nbr_dists[de.IFM] = part.part_neighbor_dist(pe.OUTP)

        # Ofmaps in gbuf can be shared by nodes with INNP.
        self.dims[de.OFM] = part.dim(pe.INPP)
        self.nbr_dists[de.OFM] = part.part_neighbor_dist(pe.INPP)

    def dim(self, dce):
        ''' Get the buffer sharing node group dimensions. '''
        return self.dims[dce]

    def size(self, dce):
        ''' Get the buffer sharing node group size. '''
        return self.dims[dce].size()

