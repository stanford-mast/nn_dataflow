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

from . import loop_blocking
from . import loop_blocking_solver
from . import partition
from . import data_category_enum as DataCategoryEnum
from . import loop_enum as LoopEnum
from . import mem_hier_enum as MemHierEnum
from . import parallel_enum as ParallelEnum
from .cost import Cost
from .data_dim_loops import DataDimLoops
from .data_layout import DataLayout
from .fmap_range import FmapPosition, FmapRange, FmapRangeMap
from .layer import Layer, InputLayer, ConvLayer, FCLayer, \
        LocalRegionLayer, PoolingLayer
from .loop_blocking_scheme import LoopBlockingScheme
from .map_strategy import MapStrategy, MapStrategyEyeriss
from .nested_loop_desc import NestedLoopDesc
from .network import Network
from .node_region import NodeRegion
from .nn_dataflow_scheme import NNDataflowScheme
from .option import Option
from .partition_scheme import PartitionScheme
from .phy_dim2 import PhyDim2
from .resource import Resource
from .scheduling import SchedulingCondition, SchedulingResult, Scheduling

from .nn_dataflow import NNDataflow

