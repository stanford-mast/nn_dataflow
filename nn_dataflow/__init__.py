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

from . import DataCategoryEnum
from . import LoopBlocking
from . import LoopBlockingSolver
from . import MemHierEnum
from . import ParallelEnum
from . import Partition
from .Cost import Cost
from .DataLayout import DataLayout
from .FmapRange import FmapPosition, FmapRange, FmapRangeMap
from .Layer import Layer, InputLayer, ConvLayer, FCLayer, \
        LocalRegionLayer, PoolingLayer
from .LoopBlockingScheme import LoopBlockingScheme
from .MapStrategy import MapStrategyEyeriss
from .NestedLoopDesc import NestedLoopDesc
from .Network import Network
from .NNDataflow import SchedulingResultDict
from .Option import Option
from .PartitionScheme import PartitionScheme
from .PhyDim2 import PhyDim2
from .Resource import NodeRegion, Resource
from .Scheduling import SchedulingCondition, SchedulingResult, Scheduling

from .NNDataflow import NNDataflow

__version__ = '1.4.0'

