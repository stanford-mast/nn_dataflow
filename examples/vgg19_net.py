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
from nn_dataflow import Layer, FCLayer

'''
VGGNet-19

Simonyan and Zisserman, 2014
'''

NN = Network('VGG19')

NN.add('conv1', Layer(3, 64, 224, 3))
NN.add('conv2', Layer(64, 64, 224, 3))

NN.add('conv3', Layer(64, 128, 112, 3))
NN.add('conv4', Layer(128, 128, 112, 3))

NN.add('conv5', Layer(128, 256, 56, 3))
NN.add('conv6', Layer(256, 256, 56, 3))
NN.add('conv7', Layer(256, 256, 56, 3))
NN.add('conv8', Layer(256, 256, 56, 3))

NN.add('conv9', Layer(256, 512, 28, 3))
NN.add('conv10', Layer(512, 512, 28, 3))
NN.add('conv11', Layer(512, 512, 28, 3))
NN.add('conv12', Layer(512, 512, 28, 3))

NN.add('conv13', Layer(512, 512, 14, 3))
NN.add('conv14', Layer(512, 512, 14, 3))
NN.add('conv15', Layer(512, 512, 14, 3))
NN.add('conv16', Layer(512, 512, 14, 3))

NN.add('fc1', FCLayer(512, 4096, 7))
NN.add('fc2', FCLayer(4096, 4096, 1))
NN.add('fc3', FCLayer(4096, 1000, 1))

