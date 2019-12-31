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

from nn_dataflow.core import MapStrategy

from . import TestMapStrategyFixture

class TestMapStrategy(TestMapStrategyFixture):
    ''' Tests for basic MapStrategy class. '''

    def setUp(self):

        super(TestMapStrategy, self).setUp()

        self.layer = self.convlayers['conv1']
        self.dim_array = self.resource['BASE'].dim_array

    def test_args(self):
        ''' Constructor arguments. '''
        ms = MapStrategy(self.layer, 4, 1, self.dim_array)

        self.assertEqual(ms.layer, self.layer)
        self.assertEqual(ms.batch_size, 4)
        self.assertEqual(ms.dim_array, self.dim_array)

    def test_inv_args(self):
        ''' Constructor arguments invalid. '''
        with self.assertRaisesRegex(TypeError, 'MapStrategy: .*layer.*'):
            _ = MapStrategy(None, 4, 1, self.dim_array)

        with self.assertRaisesRegex(ValueError, 'MapStrategy: .*occupancy.*'):
            _ = MapStrategy(self.layer, 4, -.1, self.dim_array)
        with self.assertRaisesRegex(ValueError, 'MapStrategy: .*occupancy.*'):
            _ = MapStrategy(self.layer, 4, 1.1, self.dim_array)

        with self.assertRaisesRegex(TypeError, 'MapStrategy: .*dim_array.*'):
            _ = MapStrategy(self.layer, 4, 1, None)

    def test_utilization(self):
        ''' Accessor utilization. '''
        ms = MapStrategy(self.layer, 4, 1, self.dim_array)

        with self.assertRaisesRegex(NotImplementedError, 'MapStrategy: .*'):
            _ = ms.utilization()

    def test_gen_nested_loop_desc(self):
        ''' Generator gen_nested_loop_desc. '''
        ms = MapStrategy(self.layer, 4, 1, self.dim_array)

        with self.assertRaisesRegex(NotImplementedError, 'MapStrategy: .*'):
            _ = ms.gen_nested_loop_desc()

