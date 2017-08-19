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

from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, PoolingLayer

'''
ResNet-152

He, Zhang, Ren, and Sun, 2015
'''

NN = Network('ResNet')

NN.set_input(InputLayer(3, 224))

_PREVS = None

NN.add('conv1', ConvLayer(3, 64, 112, 7, 2))
NN.add('pool1', PoolingLayer(64, 56, 2))

for i in range(1, 4):
    NN.add('conv2_{}_a'.format(i),
           ConvLayer(64, 64, 56, 1) if i == 1 else ConvLayer(256, 64, 56, 1),
           prevs=_PREVS)
    NN.add('conv2_{}_b'.format(i), ConvLayer(64, 64, 56, 3))
    NN.add('conv2_{}_c'.format(i), ConvLayer(64, 256, 56, 1))

    # With residual shortcut.
    if i == 1:
        # Residual does not cross module.
        _PREVS = None
    else:
        _PREVS = ('conv2_{}_c'.format(i), 'conv2_{}_c'.format(i - 1))

for i in range(1, 9):
    NN.add('conv3_{}_a'.format(i),
           ConvLayer(256, 128, 28, 1, 2) if i == 1
           else ConvLayer(512, 128, 28, 1),
           prevs=_PREVS)
    NN.add('conv3_{}_b'.format(i), ConvLayer(128, 128, 28, 3))
    NN.add('conv3_{}_c'.format(i), ConvLayer(128, 512, 28, 1))

    # With residual shortcut.
    if i == 1:
        # Residual does not cross module.
        _PREVS = None
    else:
        _PREVS = ('conv3_{}_c'.format(i), 'conv3_{}_c'.format(i - 1))

for i in range(1, 37):
    NN.add('conv4_{}_a'.format(i),
           ConvLayer(512, 256, 14, 1, 2) if i == 1
           else ConvLayer(1024, 256, 14, 1),
           prevs=_PREVS)
    NN.add('conv4_{}_b'.format(i), ConvLayer(256, 256, 14, 3))
    NN.add('conv4_{}_c'.format(i), ConvLayer(256, 1024, 14, 1))

    # With residual shortcut.
    if i == 1:
        # Residual does not cross module.
        _PREVS = None
    else:
        _PREVS = ('conv4_{}_c'.format(i), 'conv4_{}_c'.format(i - 1))

for i in range(1, 4):
    NN.add('conv5_{}_a'.format(i),
           ConvLayer(1024, 512, 7, 1, 2) if i == 1
           else ConvLayer(2048, 512, 7, 1),
           prevs=_PREVS)
    NN.add('conv5_{}_b'.format(i), ConvLayer(512, 512, 7, 3))
    NN.add('conv5_{}_c'.format(i), ConvLayer(512, 2048, 7, 1))

    # With residual shortcut.
    if i == 1:
        # Residual does not cross module.
        _PREVS = None
    else:
        _PREVS = ('conv5_{}_c'.format(i), 'conv5_{}_c'.format(i - 1))

