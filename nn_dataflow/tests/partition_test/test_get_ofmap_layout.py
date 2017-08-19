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

from nn_dataflow.core import partition
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PartitionScheme
from nn_dataflow.core import PhyDim2

from . import TestPartitionFixture

class TestGetOfmapLayout(TestPartitionFixture):
    ''' Tests for get_ofmap_layout function. '''

    def test_shrink_one(self):
        ''' Shrink. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (4, 4), (1, 1)))

        omr = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(4, 30),
                         type=NodeRegion.DATA)
        dl = partition.get_ofmap_layout(layer, self.batch_size, part, omr)

        self._general_assert(dl, omr, layer)

        self.assertEqual(len(list(dl.frmap.items())), 6 * 5 * (2 * 2))
        self.assertEqual(len(list(dl.frmap.items())), 4 * 30)

    def test_shrink_multi(self):
        ''' Shrink multiple. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (4, 4), (1, 1)))

        omr = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(2, 2),
                         type=NodeRegion.DATA)
        dl = partition.get_ofmap_layout(layer, self.batch_size, part, omr)

        self._general_assert(dl, omr, layer)

        self.assertEqual(len(list(dl.frmap.items())), 2 * 2)
        self.assertEqual(len(list(dl.frmap.items())), 4)

    def test_shrink_nondiv(self):
        ''' Shrink non-dividable. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (4, 4), (1, 1)))

        omr = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(3, 54),
                         type=NodeRegion.DATA)
        dl = partition.get_ofmap_layout(layer, self.batch_size, part, omr)

        self._general_assert(dl, omr, layer)

        # For height, 3 // 2 = 1.
        # For width, 54 // 5 = 10, 10 // 3 = 3.
        self.assertEqual(len(list(dl.frmap.items())), 6 * 5 * (1 * 3))
        self.assertLess(len(list(dl.frmap.items())), 3 * 54)

    def test_shrink_with_inpp(self):
        ''' Shrink with INPP. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.INPP, pe.BATP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (4, 4), (4, 4)))

        omr = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(4, 30),
                         type=NodeRegion.DATA)
        dl = partition.get_ofmap_layout(layer, self.batch_size, part, omr)

        self._general_assert(dl, omr, layer)

        self.assertEqual(len(list(dl.frmap.items())), 6 * 5 * (2 * 2))
        self.assertEqual(len(list(dl.frmap.items())), 4 * 30)

    def test_shrink_all(self):
        ''' Shrink all. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.INPP, pe.BATP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (2, 5), (4, 4), (4, 4)))

        omr = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                         type=NodeRegion.DATA)
        dl = partition.get_ofmap_layout(layer, self.batch_size, part, omr)

        self._general_assert(dl, omr, layer)

        self.assertEqual(len(list(dl.frmap.items())), 1)

    def test_extend_one(self):
        ''' Extend. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.INPP, pe.BATP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (1, 1), (1, 1)))

        omr = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(4, 30),
                         type=NodeRegion.DATA)
        dl = partition.get_ofmap_layout(layer, self.batch_size, part, omr)

        self._general_assert(dl, omr, layer)

        self.assertEqual(len(list(dl.frmap.items())), 6 * 5 * (2 * 2))
        self.assertEqual(len(list(dl.frmap.items())), 4 * 30)

    def test_extend_nondiv(self):
        ''' Extend non-dividable. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.INPP, pe.BATP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (1, 1), (1, 1)))

        omr = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(5, 40),
                         type=NodeRegion.DATA)
        dl = partition.get_ofmap_layout(layer, self.batch_size, part, omr)

        self._general_assert(dl, omr, layer)

        # For height, 5 // 2 = 2.
        # For width, 40 // (3 * 5) == 2.
        self.assertEqual(len(list(dl.frmap.items())), 6 * 5 * (2 * 2))
        self.assertLess(len(list(dl.frmap.items())), 5 * 40)

    def test_extend_with_inpp(self):
        ''' Extend with INPP. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.INPP, pe.BATP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (1, 1), (4, 4)))

        omr = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(4, 30),
                         type=NodeRegion.DATA)
        dl = partition.get_ofmap_layout(layer, self.batch_size, part, omr)

        self._general_assert(dl, omr, layer)

        self.assertEqual(len(list(dl.frmap.items())), 6 * 5 * (2 * 2))

    def test_shrink_extend(self):
        ''' Both shrink and extend. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (4, 4), (1, 1)))

        omr = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(16, 16),
                         type=NodeRegion.DATA)
        dl = partition.get_ofmap_layout(layer, self.batch_size, part, omr)

        self._general_assert(dl, omr, layer)

        self.assertLessEqual(len(list(dl.frmap.items())), 16 * 16)

    def test_zero_region(self):
        ''' Zero dim region. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (4, 4), (1, 1)))

        omr = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(0, 0),
                         type=NodeRegion.DATA)

        with self.assertRaisesRegexp(ValueError, 'partition .*empty.*'):
            _ = partition.get_ofmap_layout(layer, self.batch_size, part, omr)

    def test_origin(self):
        ''' Same dim but different origins. '''
        layer = self.layers['BASE']

        part = PartitionScheme(order=(pe.BATP, pe.INPP, pe.OUTP, pe.OFMP),
                               pdims=((2, 3), (1, 5), (4, 4), (1, 1)))

        omr1 = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(16, 16),
                          type=NodeRegion.DATA)
        dl1 = partition.get_ofmap_layout(layer, self.batch_size, part, omr1)

        omr2 = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(16, 16),
                          type=NodeRegion.DATA)
        dl2 = partition.get_ofmap_layout(layer, self.batch_size, part, omr2)

        self._general_assert(dl1, omr1, layer)
        self._general_assert(dl2, omr2, layer)

        self.assertSetEqual(set(dl1.frmap.items()), set(dl2.frmap.items()))

    def _general_assert(self, data_layout, output_mem_region, layer):
        self.assertEqual(data_layout.origin, output_mem_region.origin)
        self.assertEqual(data_layout.frmap.complete_fmap_range().size(),
                         layer.total_ofmap_size(self.batch_size))
        self.assertTrue(data_layout.is_in_region(output_mem_region))

