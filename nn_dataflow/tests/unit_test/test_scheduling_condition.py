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

import unittest

from nn_dataflow.core import DataLayout
from nn_dataflow.core import FmapRange
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource
from nn_dataflow.core import SchedulingCondition
from nn_dataflow.core import SchedulingConstraint

class TestSchedulingCondition(unittest.TestCase):
    ''' Tests for SchedulingCondition. '''

    def setUp(self):

        self.resource = Resource(
            proc_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                   type=NodeRegion.PROC),
            dram_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                   type=NodeRegion.DRAM),
            src_data_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                       type=NodeRegion.DRAM),
            dst_data_region=NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                       type=NodeRegion.DRAM),
            dim_array=PhyDim2(16, 16), size_gbuf=65536, size_regf=64,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)

        self.none_cstr = SchedulingConstraint()

        part = PartitionScheme(order=range(pe.NUM), pdims=[(1, 1)] * pe.NUM)
        self.ifmap_layout = DataLayout(frngs=(FmapRange((0, 0, 0, 0),
                                                        (2, 4, 16, 16)),),
                                       regions=(self.resource.src_data_region,),
                                       parts=(part,))

        self.sched_seq = (2, 0, 0)

    def test_valid_args(self):
        ''' Valid arguments. '''
        condition = SchedulingCondition(resource=self.resource,
                                        constraint=self.none_cstr,
                                        ifmap_layout=self.ifmap_layout,
                                        sched_seq=self.sched_seq)
        self.assertEqual(condition.resource, self.resource)
        self.assertEqual(condition.constraint, self.none_cstr)
        self.assertEqual(condition.ifmap_layout, self.ifmap_layout)
        self.assertTupleEqual(condition.sched_seq, self.sched_seq)

    def test_invalid_resource(self):
        ''' Invalid resource. '''
        with self.assertRaisesRegex(TypeError,
                                    'SchedulingCondition: .*resource.*'):
            _ = SchedulingCondition(resource=None,
                                    constraint=self.none_cstr,
                                    ifmap_layout=self.ifmap_layout,
                                    sched_seq=self.sched_seq)

    def test_invalid_constraint(self):
        ''' Invalid constraint. '''
        with self.assertRaisesRegex(TypeError,
                                    'SchedulingCondition: .*constraint.*'):
            _ = SchedulingCondition(resource=self.resource,
                                    constraint=None,
                                    ifmap_layout=self.ifmap_layout,
                                    sched_seq=self.sched_seq)

    def test_invalid_ifmap_layout(self):
        ''' Invalid ifmap_layout. '''
        with self.assertRaisesRegex(TypeError,
                                    'SchedulingCondition: .*ifmap_layout.*'):
            _ = SchedulingCondition(resource=self.resource,
                                    constraint=self.none_cstr,
                                    ifmap_layout=None,
                                    sched_seq=self.sched_seq)

    def test_invalid_sched_seq(self):
        ''' Invalid sched_seq. '''
        with self.assertRaisesRegex(TypeError,
                                    'SchedulingCondition: .*sched_seq.*'):
            _ = SchedulingCondition(resource=self.resource,
                                    constraint=self.none_cstr,
                                    ifmap_layout=self.ifmap_layout,
                                    sched_seq=list(self.sched_seq))

        with self.assertRaisesRegex(ValueError,
                                    'SchedulingCondition: .*sched_seq.*'):
            _ = SchedulingCondition(resource=self.resource,
                                    constraint=self.none_cstr,
                                    ifmap_layout=self.ifmap_layout,
                                    sched_seq=self.sched_seq[:-1])

