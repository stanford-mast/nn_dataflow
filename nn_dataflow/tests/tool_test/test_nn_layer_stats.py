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

        self.args = ['python', 'nn_dataflow/tools/nn_layer_stats.py',
                     'alex_net', '-b', '16']

    def test_default_invoke(self):
        ''' Default invoke. '''
        ret = self._call(self.args)
        self.assertEqual(ret, 0)

    def _call(self, args):
        return subprocess.call(args, cwd=self.cwd,
                               stderr=subprocess.STDOUT,
                               stdout=open(os.devnull, 'w'))

