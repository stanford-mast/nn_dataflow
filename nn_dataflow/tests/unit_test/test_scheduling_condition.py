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

import unittest

from nn_dataflow.core import DataLayout
from nn_dataflow.core import FmapRange, FmapRangeMap
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource
from nn_dataflow.core import SchedulingCondition

class TestSchedulingCondition(unittest.TestCase):
    ''' Tests for SchedulingCondition. '''

    def setUp(self):

        self.resource = Resource(
            proc_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                   type=NodeRegion.PROC),
            data_regions=(NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                     type=NodeRegion.DATA),),
            dim_array=PhyDim2(16, 16), size_gbuf=65536, size_regf=64)

        frmap = FmapRangeMap()
        frmap.add(FmapRange((0, 0, 0, 0), (2, 4, 16, 16)), (PhyDim2(0, 0),))
        self.ifmap_layout = DataLayout(origin=PhyDim2(0, 0), frmap=frmap,
                                       type=NodeRegion.DATA)

    def test_valid_args(self):
        ''' Valid arguments. '''
        condition = SchedulingCondition(resource=self.resource,
                                        ifmap_layout=self.ifmap_layout)
        self.assertEqual(condition.resource, self.resource)
        self.assertEqual(condition.ifmap_layout, self.ifmap_layout)

    def test_invalid_resource(self):
        ''' Invalid resource. '''
        with self.assertRaisesRegexp(TypeError,
                                     'SchedulingCondition: .*resource.*'):
            _ = SchedulingCondition(resource=None,
                                    ifmap_layout=self.ifmap_layout)

    def test_invalid_ifmap_layout(self):
        ''' Invalid resource. '''
        with self.assertRaisesRegexp(TypeError,
                                     'SchedulingCondition: .*ifmap_layout.*'):
            _ = SchedulingCondition(resource=self.resource,
                                    ifmap_layout=None)

