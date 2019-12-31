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

import os
import subprocess

class TestNNLayerStats(unittest.TestCase):
    ''' Tests for NN layer stats tool. '''

    def setUp(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        self.cwd = os.path.join(cwd, '..', '..', '..')
        self.assertTrue(os.path.isdir(self.cwd))
        self.assertTrue(os.path.isdir(
            os.path.join(self.cwd, 'nn_dataflow', 'tools')))

        self.args = ['python', '-m', 'nn_dataflow.tools.nn_layer_stats',
                     'alex_net', '-b', '16']

    def test_default_invoke(self):
        ''' Default invoke. '''
        ret = self._call(self.args)
        self.assertEqual(ret, 0)

    def _call(self, args):
        with open(os.devnull, 'w') as output:
            result = subprocess.call(args, cwd=self.cwd,
                                     stderr=subprocess.STDOUT,
                                     stdout=output)
        return result

