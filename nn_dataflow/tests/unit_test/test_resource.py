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

from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource

class TestResource(unittest.TestCase):
    ''' Tests for Resource. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        resource = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                   origin=PhyDim2(0, 0),
                                                   type=NodeRegion.PROC),
                            data_regions=(NodeRegion(dim=PhyDim2(2, 1),
                                                     origin=PhyDim2(0, 0),
                                                     type=NodeRegion.DATA),
                                          NodeRegion(dim=PhyDim2(2, 1),
                                                     origin=PhyDim2(0, 1),
                                                     type=NodeRegion.DATA)),
                            dim_array=PhyDim2(16, 16),
                            size_gbuf=131072,
                            size_regf=512,
                           )
        self.assertTupleEqual(resource.proc_region.dim, (2, 2), 'proc_region')
        self.assertTupleEqual(resource.dim_array, (16, 16), 'dim_array')
        self.assertEqual(resource.size_gbuf, 131072, 'size_gbuf')
        self.assertEqual(resource.size_regf, 512, 'size_regf')

    def test_invalid_proc_region(self):
        ''' Invalid proc_region. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*proc_region.*'):
            _ = Resource(proc_region=PhyDim2(2, 2),
                         data_regions=(NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 0),
                                                  type=NodeRegion.DATA),
                                       NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 1),
                                                  type=NodeRegion.DATA)),
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                        )

    def test_invalid_proc_region_data(self):
        ''' Invalid proc_region with type DATA. '''
        with self.assertRaisesRegexp(ValueError, 'Resource: .*proc_.*type.*'):
            _ = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.DATA),
                         data_regions=(NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 0),
                                                  type=NodeRegion.DATA),
                                       NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 1),
                                                  type=NodeRegion.DATA)),
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                        )

    def test_invalid_dim_array(self):
        ''' Invalid dim_array. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*dim_array.*'):
            _ = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.PROC),
                         data_regions=(NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 0),
                                                  type=NodeRegion.DATA),
                                       NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 1),
                                                  type=NodeRegion.DATA)),
                         dim_array=(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                        )

    def test_invalid_size_gbuf(self):
        ''' Invalid size_gbuf. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*size_gbuf.*'):
            _ = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.PROC),
                         data_regions=(NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 0),
                                                  type=NodeRegion.DATA),
                                       NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 1),
                                                  type=NodeRegion.DATA)),
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=(131072,),
                         size_regf=512,
                        )

    def test_invalid_size_regf(self):
        ''' Invalid size_regf. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*size_regf.*'):
            _ = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.PROC),
                         data_regions=(NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 0),
                                                  type=NodeRegion.DATA),
                                       NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 1),
                                                  type=NodeRegion.DATA)),
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=(512,),
                        )

    def test_invalid_data_regions_type(self):
        ''' Invalid data_regions type. '''
        with self.assertRaisesRegexp(TypeError, 'Resource: .*data_regions.*'):
            _ = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.PROC),
                         data_regions=[NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 0),
                                                  type=NodeRegion.DATA),],
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=(512,),
                        )
        with self.assertRaisesRegexp(TypeError, 'Resource: .*data_regions.*'):
            _ = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.PROC),
                         data_regions=NodeRegion(dim=PhyDim2(2, 1),
                                                 origin=PhyDim2(0, 0),
                                                 type=NodeRegion.DATA),
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=(512,),
                        )
        with self.assertRaisesRegexp(TypeError, 'Resource: .*data_regions.*'):
            _ = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.PROC),
                         data_regions=(NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 0),
                                                  type=NodeRegion.DATA),
                                       PhyDim2(2, 1)),
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=(512,),
                        )

    def test_invalid_data_regions_len(self):
        ''' Invalid data_regions len. '''
        with self.assertRaisesRegexp(ValueError, 'Resource: .*data_regions.*'):
            _ = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.PROC),
                         data_regions=(NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 0),
                                                  type=NodeRegion.DATA),) * 3,
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=(512,),
                        )

    def test_invalid_data_regions_proc(self):
        ''' Invalid data_regions with type PROC. '''
        with self.assertRaisesRegexp(ValueError, 'Resource: .*data_.*type.*'):
            _ = Resource(proc_region=NodeRegion(dim=PhyDim2(2, 2),
                                                origin=PhyDim2(0, 0),
                                                type=NodeRegion.PROC),
                         data_regions=(NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 0),
                                                  type=NodeRegion.PROC),
                                       NodeRegion(dim=PhyDim2(2, 1),
                                                  origin=PhyDim2(0, 1),
                                                  type=NodeRegion.DATA)),
                         dim_array=PhyDim2(16, 16),
                         size_gbuf=131072,
                         size_regf=512,
                        )

    def test_src_data_region(self):
        ''' Accessor src_data_region. '''
        nr1 = NodeRegion(dim=PhyDim2(2, 1), origin=PhyDim2(0, 0),
                         type=NodeRegion.DATA)
        nr2 = NodeRegion(dim=PhyDim2(2, 1), origin=PhyDim2(0, 1),
                         type=NodeRegion.DATA)
        resource = Resource(proc_region=NodeRegion(dim=PhyDim2(4, 4),
                                                   origin=PhyDim2(0, 0),
                                                   type=NodeRegion.PROC),
                            dim_array=PhyDim2(16, 16),
                            size_gbuf=131072,
                            size_regf=512,
                            data_regions=(nr1, nr2))
        self.assertEqual(resource.src_data_region(), nr1, 'src_data_region')
        resource = Resource(proc_region=NodeRegion(dim=PhyDim2(4, 4),
                                                   origin=PhyDim2(0, 0),
                                                   type=NodeRegion.PROC),
                            dim_array=PhyDim2(16, 16),
                            size_gbuf=131072,
                            size_regf=512,
                            data_regions=(nr1,))
        self.assertEqual(resource.src_data_region(), nr1, 'src_data_region')

    def test_dst_data_region(self):
        ''' Accessor dst_data_region. '''
        nr1 = NodeRegion(dim=PhyDim2(2, 1), origin=PhyDim2(0, 0),
                         type=NodeRegion.DATA)
        nr2 = NodeRegion(dim=PhyDim2(2, 1), origin=PhyDim2(0, 1),
                         type=NodeRegion.DATA)
        resource = Resource(proc_region=NodeRegion(dim=PhyDim2(4, 4),
                                                   origin=PhyDim2(0, 0),
                                                   type=NodeRegion.PROC),
                            dim_array=PhyDim2(16, 16),
                            size_gbuf=131072,
                            size_regf=512,
                            data_regions=(nr1, nr2))
        self.assertEqual(resource.dst_data_region(), nr2, 'dst_data_region')
        resource = Resource(proc_region=NodeRegion(dim=PhyDim2(4, 4),
                                                   origin=PhyDim2(0, 0),
                                                   type=NodeRegion.PROC),
                            dim_array=PhyDim2(16, 16),
                            size_gbuf=131072,
                            size_regf=512,
                            data_regions=(nr1,))
        self.assertEqual(resource.dst_data_region(), nr1, 'dst_data_region')

