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

from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer

'''
VGGNet-19

Simonyan and Zisserman, 2014
'''

NN = Network('VGG19')

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1', ConvLayer(3, 64, 224, 3))
NN.add('conv2', ConvLayer(64, 64, 224, 3))
NN.add('pool1', PoolingLayer(64, 112, 2))

NN.add('conv3', ConvLayer(64, 128, 112, 3))
NN.add('conv4', ConvLayer(128, 128, 112, 3))
NN.add('pool2', PoolingLayer(128, 56, 2))

NN.add('conv5', ConvLayer(128, 256, 56, 3))
NN.add('conv6', ConvLayer(256, 256, 56, 3))
NN.add('conv7', ConvLayer(256, 256, 56, 3))
NN.add('conv8', ConvLayer(256, 256, 56, 3))
NN.add('pool3', PoolingLayer(256, 28, 2))

NN.add('conv9', ConvLayer(256, 512, 28, 3))
NN.add('conv10', ConvLayer(512, 512, 28, 3))
NN.add('conv11', ConvLayer(512, 512, 28, 3))
NN.add('conv12', ConvLayer(512, 512, 28, 3))
NN.add('pool4', PoolingLayer(512, 14, 2))

NN.add('conv13', ConvLayer(512, 512, 14, 3))
NN.add('conv14', ConvLayer(512, 512, 14, 3))
NN.add('conv15', ConvLayer(512, 512, 14, 3))
NN.add('conv16', ConvLayer(512, 512, 14, 3))
NN.add('pool5', PoolingLayer(512, 7, 2))

NN.add('fc1', FCLayer(512, 4096, 7))
NN.add('fc2', FCLayer(4096, 4096))
NN.add('fc3', FCLayer(4096, 1000))

