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

from collections import OrderedDict

from nn_dataflow import Layer, FCLayer

'''
AlexNet

Used in Eyeriss paper.
'''

LAYERS = OrderedDict()

LAYERS['conv1'] = Layer(3, 96, 55, 11, 4)
LAYERS['conv2'] = Layer(48, 256, 27, 5)
LAYERS['conv3'] = Layer(256, 384, 13, 3)
LAYERS['conv4'] = Layer(192, 384, 13, 3)
LAYERS['conv5'] = Layer(192, 256, 13, 3)
LAYERS['fc1'] = FCLayer(256, 4096, 6)
LAYERS['fc2'] = FCLayer(4096, 4096, 1)
LAYERS['fc3'] = FCLayer(4096, 1000, 1)

