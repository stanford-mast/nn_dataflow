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
nn_dataflow module.
'''

from . import DataCategoryEnum
from . import LoopBlocking
from . import MemHierEnum
from . import ParallelEnum
from . import Partition
from . import Solver
from .Cost import Cost
from .Layer import Layer, FCLayer
from .LoopBlockingScheme import LoopBlockingScheme
from .MapStrategy import MapStrategyEyeriss
from .NestedLoopDesc import NestedLoopDesc
from .Option import Option
from .Partition import Partition2dScheme
from .PhyDim2 import PhyDim2
from .Resource import Resource

from .Schedule import schedule_search

