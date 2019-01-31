""" $lic$
Copyright (C) 2016-2019 by The Board of Trustees of Stanford University

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

from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource

class TestResource(unittest.TestCase):
    ''' Tests for Resource. '''

    def setUp(self):
        self.proc_region = NodeRegion(dim=PhyDim2(2, 2), origin=PhyDim2(0, 0),
                                      type=NodeRegion.PROC)
        self.dram_region = NodeRegion(dim=PhyDim2(2, 2), origin=PhyDim2(0, 0),
                                      type=NodeRegion.DRAM)
        self.src_data_region = NodeRegion(dim=PhyDim2(2, 1),
                                          origin=PhyDim2(0, 0),
                                          type=NodeRegion.DRAM)
        self.dst_data_region = NodeRegion(dim=PhyDim2(2, 1),
                                          origin=PhyDim2(0, 1),
                                          type=NodeRegion.DRAM)

    def test_valid_args(self):
        ''' Valid arguments. '''
        resource = Resource(proc_region=self.proc_region,
                            dram_region=self.dram_region,
                            src_data_region=self.src_data_region,
                            dst_data_region=self.dst_data_region,
                            dim_array=PhyDim2(16, 16),
                            size_gbuf=131072,
                            size_regf=512,
                            array_bus_width=8,
                            dram_bandwidth=128,
                           )
        self.assertTupleEqual(resource.proc_region.dim, (2, 2), 'proc_region')
        self.assertTupleEqual(resource.dram_region.dim, (2, 2), 'dram_region')
        self.assertTupleEqual(resource.dim_array, (16, 16), 'dim_array')
        self.assertEqual(resource.size_gbuf, 131072, 'size_gbuf')
        self.assertEqual(resource.size_regf, 512, 'size_regf')
        self.assertEqual(resource.array_bus_width, 8, 'array_bus_width')
        self.assertEqual(resource.dram_bandwidth, 128, 'dram_bandwidth')

    def test_invalid_proc_region(self):
        ''' Invalid proc_region. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*proc_region.*'):
            _ = Resource(proc_region=PhyDim2(2, 2),
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=128,
                        )

    def test_invalid_proc_region_dram(self):
        ''' Invalid proc_region with type DRAM. '''
        with self.assertRaisesRegexp(ValueError, 'Resource: .*proc_.*type.*'):
            _ = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.DRAM),
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=128,
                        )

    def test_invalid_dram_region(self):
        ''' Invalid dram_region. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*dram_region.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=PhyDim2(2, 2),
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=128,
                        )

    def test_invalid_dram_region_proc(self):
        ''' Invalid dram_region with type DRAM. '''
        with self.assertRaisesRegexp(ValueError, 'Resource: .*dram_.*type.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.PROC),
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=128,
                        )

    def test_invalid_data_region(self):
        ''' Invalid src/dst_proc_region. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*src_data_.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=PhyDim2(2, 1),
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=128,
                        )
        with self.assertRaisesRegexp(TypeError, 'Resource: .*dst_data_.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=PhyDim2(2, 1),
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=128,
                        )

    def test_invalid_dim_array(self):
        ''' Invalid dim_array. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*dim_array.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=128,
                        )

    def test_invalid_size_gbuf(self):
        ''' Invalid size_gbuf. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*size_gbuf.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=(131072,),
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=128,
                        )

    def test_invalid_size_regf(self):
        ''' Invalid size_regf. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*size_regf.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=(512,),
                         array_bus_width=8,
                         dram_bandwidth=128,
                        )

    def test_invalid_array_bus_width(self):
        ''' Invalid array_bus_width. '''
        with self.assertRaisesRegexp(TypeError,
                                     'Resource: .*array_bus_width.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=1.2,
                         dram_bandwidth=128,
                        )
        with self.assertRaisesRegexp(ValueError,
                                     'Resource: .*array_bus_width.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=-2,
                         dram_bandwidth=128,
                        )
        with self.assertRaisesRegexp(ValueError,
                                     'Resource: .*array_bus_width.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=0,
                         dram_bandwidth=128,
                        )

    def test_invalid_dram_bandwidth(self):
        ''' Invalid dram_bandwidth. '''
        with self.assertRaisesRegexp(TypeError,
                                     'Resource: .*dram_bandwidth.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=None,
                        )
        with self.assertRaisesRegexp(ValueError,
                                     'Resource: .*dram_bandwidth.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=-3,
                        )
        with self.assertRaisesRegexp(ValueError,
                                     'Resource: .*dram_bandwidth.*'):
            _ = Resource(proc_region=self.proc_region,
                         dram_region=self.dram_region,
                         src_data_region=self.src_data_region,
                         dst_data_region=self.dst_data_region,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                         array_bus_width=8,
                         dram_bandwidth=0,
                        )

