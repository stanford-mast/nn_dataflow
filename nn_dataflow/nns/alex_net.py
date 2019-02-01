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

from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer

'''
AlexNet

Krizhevsky, Sutskever, and Hinton, 2012
'''

NN = Network('AlexNet')

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1_a', ConvLayer(3, 48, 55, 11, 4), prevs=(NN.INPUT_LAYER_KEY,))
NN.add('conv1_b', ConvLayer(3, 48, 55, 11, 4), prevs=(NN.INPUT_LAYER_KEY,))
NN.add('pool1_a', PoolingLayer(48, 27, 3, strd=2), prevs=('conv1_a',))
NN.add('pool1_b', PoolingLayer(48, 27, 3, strd=2), prevs=('conv1_b',))
# Norm layer is ignored.

NN.add('conv2_a', ConvLayer(48, 128, 27, 5), prevs=('pool1_a',))
NN.add('conv2_b', ConvLayer(48, 128, 27, 5), prevs=('pool1_b',))
NN.add('pool2_a', PoolingLayer(128, 13, 3, strd=2), prevs=('conv2_a',))
NN.add('pool2_b', PoolingLayer(128, 13, 3, strd=2), prevs=('conv2_b',))
# Norm layer is ignored.

NN.add('conv3_a', ConvLayer(256, 192, 13, 3), prevs=('pool2_a', 'pool2_b'))
NN.add('conv3_b', ConvLayer(256, 192, 13, 3), prevs=('pool2_a', 'pool2_b'))
NN.add('conv4_a', ConvLayer(192, 192, 13, 3), prevs=('conv3_a',))
NN.add('conv4_b', ConvLayer(192, 192, 13, 3), prevs=('conv3_b',))
NN.add('conv5_a', ConvLayer(192, 128, 13, 3), prevs=('conv4_a',))
NN.add('conv5_b', ConvLayer(192, 128, 13, 3), prevs=('conv4_b',))
NN.add('pool3_a', PoolingLayer(128, 6, 3, strd=2), prevs=('conv5_a',))
NN.add('pool3_b', PoolingLayer(128, 6, 3, strd=2), prevs=('conv5_b',))

NN.add('fc1', FCLayer(256, 4096, 6), prevs=('pool3_a', 'pool3_b'))
NN.add('fc2', FCLayer(4096, 4096))
NN.add('fc3', FCLayer(4096, 1000))

