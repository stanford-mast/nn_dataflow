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

from nn_dataflow.core import Cost
from nn_dataflow.core import MemHierEnum as me

class TestCost(unittest.TestCase):
    ''' Tests for Cost. '''

    def test_valid_args(self):
        ''' Valid arguments. '''
        cost = Cost(mac_op=1,
                    mem_hier=(200, 6, 2, 1),
                    noc_hop=10,
                    unit_static=0,
                   )
        self.assertEqual(cost.mac_op, 1, 'mac_op')
        self.assertEqual(cost.mem_hier, (200, 6, 2, 1), 'mem_hier')
        self.assertEqual(cost.noc_hop, 10, 'noc_hop')
        self.assertEqual(cost.unit_static, 0, 'unit_static')

    def test_invalid_mac_op(self):
        ''' Invalid mac_op. '''
        with self.assertRaisesRegexp(TypeError, 'Cost: .*mac_op.*'):
            _ = Cost(mac_op=(1, 2),
                     mem_hier=(200, 6, 2, 1),
                     noc_hop=10,
                     unit_static=0,
                    )

    def test_invalid_mem_hier_type(self):
        ''' Invalid mem_hier type. '''
        with self.assertRaisesRegexp(TypeError, 'Cost: .*mem_hier.*'):
            _ = Cost(mac_op=1,
                     mem_hier=200,
                     noc_hop=10,
                     unit_static=0,
                    )
        with self.assertRaisesRegexp(TypeError, 'Cost: .*mem_hier.*'):
            _ = Cost(mac_op=1,
                     mem_hier=[200, 6, 2, 1],
                     noc_hop=10,
                     unit_static=0,
                    )

    def test_invalid_mem_hier_len(self):
        ''' Invalid mem_hier len. '''
        with self.assertRaisesRegexp(ValueError, 'Cost: .*mem_hier.*'):
            _ = Cost(mac_op=1,
                     mem_hier=(200, 6),
                     noc_hop=10,
                     unit_static=0,
                    )

    def test_invalid_noc_hop(self):
        ''' Invalid noc_hop. '''
        with self.assertRaisesRegexp(TypeError, 'Cost: .*noc_hop.*'):
            _ = Cost(mac_op=1,
                     mem_hier=(200, 6, 2, 1),
                     noc_hop=[10, 10],
                     unit_static=0,
                    )

    def test_invalid_unit_static(self):
        ''' Invalid unit_static. '''
        with self.assertRaisesRegexp(TypeError, 'Cost: .*unit_static.*'):
            _ = Cost(mac_op=1,
                     mem_hier=(200, 6, 2, 1),
                     noc_hop=10,
                     unit_static=set([1, 2]),
                    )

    def test_mem_hier_at(self):
        ''' Accessor mem_hier. '''
        cost = Cost(mac_op=1,
                    mem_hier=(200, 6, 2, 1),
                    noc_hop=10,
                    unit_static=0,
                   )
        self.assertEqual(cost.mem_hier_at(me.DRAM), 200, 'mem_hier: DRAM')
        self.assertEqual(cost.mem_hier_at(me.GBUF), 6, 'mem_hier: GBUF')
        self.assertEqual(cost.mem_hier_at(me.ITCN), 2, 'mem_hier: ITCN')
        self.assertEqual(cost.mem_hier_at(me.REGF), 1, 'mem_hier: REGF')

    def test_mem_hier_at_error(self):
        ''' Accessor mem_hier error. '''
        cost = Cost(mac_op=1,
                    mem_hier=(200, 6, 2, 1),
                    noc_hop=10,
                    unit_static=0,
                   )
        self.assertIsNone(cost.mem_hier_at(me.NUM))
        self.assertIsNone(cost.mem_hier_at(None))

