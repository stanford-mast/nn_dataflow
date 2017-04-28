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

from nn_dataflow import Network
from nn_dataflow import Layer

'''
ResNet-152

He, Zhang, Ren, and Sun, 2015
'''

NN = Network('ResNet')

_PREVS = None

NN.add('conv1', Layer(3, 64, 112, 7, 2))

for i in range(1, 4):
    NN.add('conv2_{}_a'.format(i),
           Layer(64, 64, 56, 1, 2) if i == 1 else Layer(256, 64, 56, 1),
           prevs=_PREVS)
    NN.add('conv2_{}_b'.format(i), Layer(64, 64, 56, 3))
    NN.add('conv2_{}_c'.format(i), Layer(64, 256, 56, 1))

    if i > 1:
        # With residual shortcut.
        _PREVS = ('conv2_{}_c'.format(i), 'conv2_{}_c'.format(i - 1))

for i in range(1, 9):
    NN.add('conv3_{}_a'.format(i),
           Layer(256, 128, 28, 1, 2) if i == 1 else Layer(512, 128, 28, 1),
           prevs=_PREVS)
    NN.add('conv3_{}_b'.format(i), Layer(128, 128, 28, 3))
    NN.add('conv3_{}_c'.format(i), Layer(128, 512, 28, 1))

    if i == 1:
        # Residual does not cross module.
        _PREVS = None
    else:
        # With residual shortcut.
        _PREVS = ('conv3_{}_c'.format(i), 'conv3_{}_c'.format(i - 1))

for i in range(1, 37):
    NN.add('conv4_{}_a'.format(i),
           Layer(512, 256, 14, 1, 2) if i == 1 else Layer(1024, 256, 14, 1),
           prevs=_PREVS)
    NN.add('conv4_{}_b'.format(i), Layer(256, 256, 14, 3))
    NN.add('conv4_{}_c'.format(i), Layer(256, 1024, 14, 1))

    if i == 1:
        # Residual does not cross module.
        _PREVS = None
    else:
        # With residual shortcut.
        _PREVS = ('conv4_{}_c'.format(i), 'conv4_{}_c'.format(i - 1))

for i in range(1, 4):
    NN.add('conv5_{}_a'.format(i),
           Layer(1024, 512, 7, 1, 2) if i == 1 else Layer(2048, 512, 7, 1),
           prevs=_PREVS)
    NN.add('conv5_{}_b'.format(i), Layer(512, 512, 7, 3))
    NN.add('conv5_{}_c'.format(i), Layer(512, 2048, 7, 1))

    if i == 1:
        # Residual does not cross module.
        _PREVS = None
    else:
        # With residual shortcut.
        _PREVS = ('conv5_{}_c'.format(i), 'conv5_{}_c'.format(i - 1))

