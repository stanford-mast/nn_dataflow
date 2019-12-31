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

class TestDataLayout(unittest.TestCase):
    ''' Tests for DataLayout. '''

    def setUp(self):
        self.frng1 = FmapRange((0, 0, 0, 0), (4, 4, 16, 16))
        self.region1 = NodeRegion(dim=PhyDim2(2, 2), origin=PhyDim2(1, 1),
                                  type=NodeRegion.DRAM)
        self.part1 = PartitionScheme(order=range(pe.NUM),
                                     pdims=(PhyDim2(1, 1), PhyDim2(2, 1),
                                            PhyDim2(1, 2), PhyDim2(1, 1)))

        self.frng2 = FmapRange((0, 0, 0, 0), (4, 3, 16, 16))
        self.region2 = NodeRegion(dim=PhyDim2(2, 1), origin=PhyDim2(0, 0),
                                  type=NodeRegion.DRAM)
        self.part2 = PartitionScheme(order=range(pe.NUM),
                                     pdims=(PhyDim2(2, 1), PhyDim2(1, 1),
                                            PhyDim2(1, 1), PhyDim2(1, 1)))

        self.dl1 = DataLayout(frngs=(self.frng1,),
                              regions=(self.region1,),
                              parts=(self.part1,))
        self.dl2 = DataLayout(frngs=(self.frng2,),
                              regions=(self.region2,),
                              parts=(self.part2,))

    def test_invalid_args(self):
        ''' Invalid arguments. '''
        with self.assertRaisesRegex(TypeError, 'DataLayout: .*frngs.*'):
            _ = DataLayout(frngs=None,
                           regions=(self.region1,),
                           parts=(self.part1,))
        with self.assertRaisesRegex(TypeError,
                                    'DataLayout: .*elements in frngs.*'):
            _ = DataLayout(frngs=(None,),
                           regions=(self.region1,),
                           parts=(self.part1,))

        with self.assertRaisesRegex(TypeError, 'DataLayout: .*regions.*'):
            _ = DataLayout(frngs=(self.frng1,),
                           regions=None,
                           parts=(self.part1,))
        with self.assertRaisesRegex(TypeError,
                                    'DataLayout: .*elements in regions.*'):
            _ = DataLayout(frngs=(self.frng1,),
                           regions=self.region1,
                           parts=(self.part1,))

        with self.assertRaisesRegex(TypeError, 'DataLayout: .*parts.*'):
            _ = DataLayout(frngs=(self.frng1,),
                           regions=(self.region1,),
                           parts=None)
        with self.assertRaisesRegex(TypeError,
                                    'DataLayout: .*elements in parts.*'):
            _ = DataLayout(frngs=(self.frng1,),
                           regions=(self.region1,),
                           parts=self.part1)

    def test_invalid_frngs(self):
        ''' Invalid frngs. '''
        # No frngs.
        with self.assertRaisesRegex(ValueError, 'DataLayout: .*frng.*'):
            _ = DataLayout(frngs=tuple(),
                           regions=(self.region1,),
                           parts=(self.part1,))

        # Not begin at 0.
        with self.assertRaisesRegex(ValueError, 'DataLayout: .*frng.*'):
            _ = DataLayout(frngs=(FmapRange((0, 4, 0, 0), (4, 8, 16, 16)),
                                  self.frng1),
                           regions=(self.region1, self.region2),
                           parts=(self.part1, self.part2))

        # b, h, w mismatch.
        with self.assertRaisesRegex(ValueError, 'DataLayout: .*frng.*'):
            _ = DataLayout(frngs=(self.frng1,
                                  FmapRange((0, 4, 0, 0), (3, 8, 16, 16))),
                           regions=(self.region1, self.region2),
                           parts=(self.part1, self.part2))
        with self.assertRaisesRegex(ValueError, 'DataLayout: .*frng.*'):
            _ = DataLayout(frngs=(self.frng1,
                                  FmapRange((0, 4, 0, 0), (4, 8, 12, 16))),
                           regions=(self.region1, self.region2),
                           parts=(self.part1, self.part2))
        with self.assertRaisesRegex(ValueError, 'DataLayout: .*frng.*'):
            _ = DataLayout(frngs=(self.frng1,
                                  FmapRange((0, 4, 0, 0), (4, 8, 16, 12))),
                           regions=(self.region1, self.region2),
                           parts=(self.part1, self.part2))

        # n discontinuous.
        with self.assertRaisesRegex(ValueError, 'DataLayout: .*frng.*'):
            _ = DataLayout(frngs=(self.frng1,
                                  FmapRange((0, 5, 0, 0), (4, 8, 16, 16))),
                           regions=(self.region1, self.region2),
                           parts=(self.part1, self.part2))

    def test_invalid_parts(self):
        ''' Invalid parts. '''
        with self.assertRaisesRegex(ValueError, 'DataLayout: .*part.*'):
            _ = DataLayout(frngs=(self.frng1,),
                           regions=(self.region1,),
                           parts=(PartitionScheme(order=range(pe.NUM),
                                                  pdims=(PhyDim2(1, 1),
                                                         PhyDim2(1, 2),
                                                         PhyDim2(1, 1),
                                                         PhyDim2(2, 1))),))

        with self.assertRaisesRegex(ValueError, 'DataLayout: .*part.*'):
            _ = DataLayout(frngs=(self.frng1,
                                  FmapRange((0, 4, 0, 0), (4, 8, 16, 16))),
                           regions=(self.region2, self.region1),
                           parts=(self.part1, self.part2))

    def test_invalid_args_length(self):
        ''' Invalid args length. '''
        with self.assertRaisesRegex(ValueError, 'DataLayout: .*length.*'):
            _ = DataLayout(frngs=(self.frng1,),
                           regions=(self.region1, self.region2),
                           parts=(self.part1,))

    def test_complete_fmap_range(self):
        ''' Get complete_fmap_range. '''
        dl = DataLayout(frngs=(self.frng1,
                               FmapRange((0, 4, 0, 0), (4, 8, 16, 16))),
                        regions=(self.region1, self.region2),
                        parts=(self.part1, self.part2))
        self.assertEqual(dl.complete_fmap_range(),
                         FmapRange((0, 0, 0, 0), (4, 8, 16, 16)))

    def test_fmap_range_map(self):
        ''' Get fmap_range_map. '''
        dl = DataLayout(frngs=(self.frng1,
                               FmapRange((0, 4, 0, 0), (4, 8, 16, 16))),
                        regions=(self.region1, self.region2),
                        parts=(self.part1, self.part2))
        frmap = dl.fmap_range_map()

        self.assertEqual(frmap.complete_fmap_range(), dl.complete_fmap_range())
        self.assertSetEqual(
            set(frmap.items()),
            {(FmapRange((0, 0, 0, 0), (2, 4, 8, 16)), PhyDim2(1, 1)),
             (FmapRange((2, 0, 0, 0), (4, 4, 8, 16)), PhyDim2(1, 2)),
             (FmapRange((0, 0, 8, 0), (2, 4, 16, 16)), PhyDim2(2, 1)),
             (FmapRange((2, 0, 8, 0), (4, 4, 16, 16)), PhyDim2(2, 2)),
             (FmapRange((0, 4, 0, 0), (4, 6, 16, 16)), PhyDim2(0, 0)),
             (FmapRange((0, 6, 0, 0), (4, 8, 16, 16)), PhyDim2(1, 0))})

    def test_nhops_to(self):
        ''' Get nhops_to. '''
        fr = FmapRange((0,) * 4, (4, 4, 16, 16))
        nhops = 2 * 4 * 8 * 16 * (5 + 6 + 6 + 7)
        self.assertEqual(self.dl1.nhops_to(fr, PhyDim2(-1, -2)), nhops)

        frng1 = FmapRange((0, 4, 0, 0), (4, 8, 16, 16))
        dl = DataLayout(frngs=(self.frng1, frng1),
                        regions=(self.region1, self.region2),
                        parts=(self.part1, self.part2))
        self.assertEqual(dl.nhops_to(fr, PhyDim2(-1, -2)), nhops)

        fr = FmapRange((0,) * 4, (16,) * 4)
        nhops += 2 * 4 * 16 * 16 * (3 + 4)
        self.assertEqual(dl.nhops_to(fr, PhyDim2(-1, -2)), nhops)

    def test_nhops_to_multidests(self):
        ''' Get nhops_to multiple destinations. '''
        fr = FmapRange((0,) * 4, (4, 4, 16, 16))
        nhops = 2 * 4 * 8 * 16 * (5 + 6 + 6 + 7) \
                + 2 * 4 * 8 * 16 * (7 + 8 + 8 + 9) \
                + 2 * 4 * 8 * 16 * (2 + 1 + 1 + 0)
        self.assertEqual(self.dl1.nhops_to(fr,
                                           PhyDim2(-1, -2), PhyDim2(-2, -3),
                                           PhyDim2(2, 2)),
                         nhops)

        frng1 = FmapRange((0, 4, 0, 0), (4, 8, 16, 16))
        dl = DataLayout(frngs=(self.frng1, frng1),
                        regions=(self.region1, self.region2),
                        parts=(self.part1, self.part2))
        self.assertEqual(dl.nhops_to(fr,
                                     PhyDim2(-1, -2), PhyDim2(-2, -3),
                                     PhyDim2(2, 2)),
                         nhops)

        fr = FmapRange((0,) * 4, (16,) * 4)
        nhops += 2 * 4 * 16 * 16 * ((3 + 4) + (5 + 6) + (4 + 3))
        self.assertEqual(dl.nhops_to(fr,
                                     PhyDim2(-1, -2), PhyDim2(-2, -3),
                                     PhyDim2(2, 2)),
                         nhops)

    def test_nhops_to_multidests_fwd(self):
        ''' Get nhops_to multiple destinations forwarding. '''
        fr = FmapRange((0,) * 4, (4, 4, 16, 16))
        # First to (2, 2), then (2, 2) to (-1, -2), (-1, -2) to (-2, -3).
        nhops = 2 * 4 * 8 * 16 * (2 + 1 + 1 + 0) \
                + 2 * 4 * 8 * 16 * (4 * 7) \
                + 2 * 4 * 8 * 16 * (4 * 2)
        self.assertEqual(self.dl1.nhops_to(fr,
                                           PhyDim2(-1, -2), PhyDim2(-2, -3),
                                           PhyDim2(2, 2),
                                           forwarding=True),
                         nhops)

        frng1 = FmapRange((0, 4, 0, 0), (4, 8, 16, 16))
        dl = DataLayout(frngs=(self.frng1, frng1),
                        regions=(self.region1, self.region2),
                        parts=(self.part1, self.part2))
        self.assertEqual(dl.nhops_to(fr,
                                     PhyDim2(-1, -2), PhyDim2(-2, -3),
                                     PhyDim2(2, 2),
                                     forwarding=True),
                         nhops)

        nhops += 2 * 4 * 16 * 16 * ((3 + 4) + 2 * 7 + 2 * 2)
        fr = FmapRange((0,) * 4, (16,) * 4)
        self.assertEqual(dl.nhops_to(fr,
                                     PhyDim2(-1, -2), PhyDim2(-2, -3),
                                     PhyDim2(2, 2),
                                     forwarding=True),
                         nhops)

        # (2, 2) to (3, 10) and (8, 4)
        nhops += 4 * 8 * 16 * 16 * (9 + 8)
        self.assertEqual(dl.nhops_to(fr,
                                     PhyDim2(-1, -2), PhyDim2(-2, -3),
                                     PhyDim2(2, 2), PhyDim2(3, 10),
                                     PhyDim2(8, 4),
                                     forwarding=True),
                         nhops)

    def test_nhops_to_invalid_kwargs(self):
        ''' Get nhops_to invalid kwargs. '''
        fr = FmapRange((0,) * 4, (4, 4, 16, 16))
        with self.assertRaisesRegex(ValueError, 'DataLayout: .*keyword.*'):
            _ = self.dl1.nhops_to(fr, PhyDim2(1, 1), f=True)

    def test_is_in(self):
        ''' Whether is_in. '''
        nr1 = self.region1
        self.assertTrue(self.dl1.is_in(nr1))

        # Extend dim.
        nr2 = NodeRegion(dim=PhyDim2(5, 5),
                         origin=nr1.origin, type=nr1.type)
        self.assertTrue(self.dl1.is_in(nr2))

        # Move origin.
        nr3 = NodeRegion(origin=PhyDim2(0, 0),
                         dim=nr2.dim, type=nr2.type)
        self.assertTrue(self.dl1.is_in(nr3))

        # Change type, not in.
        nr4 = NodeRegion(type=NodeRegion.PROC,
                         origin=nr3.origin, dim=nr3.dim)
        self.assertFalse(self.dl1.is_in(nr4))

        # Move origin to not containing.
        nr5 = NodeRegion(origin=PhyDim2(0, 0),
                         dim=nr1.dim, type=nr1.type)
        self.assertFalse(self.dl1.is_in(nr5))

        # Multi-cover.
        nr6_1 = NodeRegion(origin=PhyDim2(1, 1), dim=PhyDim2(2, 1),
                           type=nr1.type)
        nr6_2 = NodeRegion(origin=PhyDim2(1, 2), dim=PhyDim2(2, 1),
                           type=nr1.type)
        self.assertTrue(self.dl1.is_in(nr6_1, nr6_2))

        # Multiple fmap ranges.
        frng1 = FmapRange((0, 4, 0, 0), (4, 8, 16, 16))
        dl = DataLayout(frngs=(self.frng1, frng1),
                        regions=(self.region1, self.region2),
                        parts=(self.part1, self.part2))
        self.assertFalse(dl.is_in(self.region1))
        self.assertTrue(dl.is_in(self.region1, self.region2))
        self.assertTrue(dl.is_in(NodeRegion(origin=PhyDim2(0, 0),
                                            dim=PhyDim2(50, 50),
                                            type=self.region1.type)))

    def test_is_in_folded(self):
        ''' Whether is_in with folded regions. '''
        # (1, 1/2), (2/3, 0/1/2), (4, 1/2)
        nr1 = NodeRegion(origin=PhyDim2(1, 1), dim=PhyDim2(1, 10),
                         type=self.region1.type, wtot=3, wbeg=2)
        # (1, 1/2), (2, 2)
        nr2 = NodeRegion(origin=PhyDim2(1, 1), dim=PhyDim2(1, 3),
                         type=self.region1.type, wtot=3, wbeg=2)
        self.assertTrue(self.dl1.is_in(nr1))
        self.assertFalse(self.dl1.is_in(nr2))

        # (1-2, 2), (3-4/5-6/7-8, 0/1/2)
        region = NodeRegion(origin=PhyDim2(1, 2), dim=PhyDim2(2, 10),
                            type=self.region1.type, wtot=3, wbeg=1)
        part = PartitionScheme(order=range(pe.NUM),
                               pdims=(PhyDim2(1, 5), PhyDim2(2, 1),
                                      PhyDim2(1, 2), PhyDim2(1, 1)))
        dl = DataLayout(frngs=self.dl1.frngs,
                        regions=(region,), parts=(part,))
        # (1-2, 1/2), (3-4/5-6, -1/0/1/2), (7-8, 0/1/2)
        nr3 = NodeRegion(origin=PhyDim2(1, 1), dim=PhyDim2(2, 13),
                         type=self.region1.type, wtot=4, wbeg=2)
        self.assertTrue(dl.is_in(nr3))
        self.assertFalse(dl.is_in(nr2))

    def test_concat(self):
        ''' Concat. '''
        fr = FmapRange((0,) * 4, (30,) * 4)

        dl = DataLayout.concat(self.dl1, self.dl2)
        self.assertEqual(dl.complete_fmap_range(),
                         FmapRange((0, 0, 0, 0), (4, 7, 16, 16)))
        self.assertEqual(dl.complete_fmap_range().size(),
                         self.dl1.complete_fmap_range().size()
                         + self.dl2.complete_fmap_range().size())
        self.assertEqual(dl.nhops_to(fr, PhyDim2(0, 0)),
                         self.dl1.nhops_to(fr, PhyDim2(0, 0))
                         + self.dl2.nhops_to(fr, PhyDim2(0, 0)))

        dl_ = DataLayout.concat(self.dl2, self.dl1)
        self.assertEqual(dl.complete_fmap_range(),
                         dl_.complete_fmap_range())
        self.assertEqual(dl.nhops_to(fr, PhyDim2(0, 0)),
                         dl_.nhops_to(fr, PhyDim2(0, 0)))

    def test_concat_invalid_type(self):
        ''' Concat invalid type. '''
        with self.assertRaisesRegex(TypeError, 'DataLayout: .*concat.*'):
            _ = DataLayout.concat(self.dl1, self.frng1)
        with self.assertRaisesRegex(TypeError, 'DataLayout: .*concat.*'):
            _ = DataLayout.concat(self.dl1, PhyDim2(1, 3))

    def test_concat_unmatch(self):
        ''' Concat unmatch. '''
        for fr in [FmapRange((0,) * 4, (4, 4, 10, 16)),
                   FmapRange((0,) * 4, (4, 4, 16, 32)),
                   FmapRange((0,) * 4, (3, 4, 16, 16))]:
            dl = DataLayout(frngs=(fr,), regions=(self.region1,),
                            parts=(self.part1,))

            with self.assertRaisesRegex(ValueError, 'DataLayout: .*match.*'):
                _ = DataLayout.concat(self.dl1, dl)

