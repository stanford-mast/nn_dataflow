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

from nn_dataflow import Layer

'''
ResNet-152

He, Zhang, Ren, and Sun, 2015
'''

LAYERS = OrderedDict()

LAYERS['conv1'] = Layer(3, 64, 112, 7, 2)

for i in range(1, 4):
    if i == 1:
        LAYERS['conv2_' + str(i) + '_a'] = Layer(64, 64, 56, 1)
    else:
        LAYERS['conv2_' + str(i) + '_a'] = Layer(256, 64, 56, 1)
    LAYERS['conv2_' + str(i) + '_b'] = Layer(64, 64, 56, 3)
    LAYERS['conv2_' + str(i) + '_c'] = Layer(64, 256, 56, 1)

for i in range(1, 9):
    if i == 1:
        LAYERS['conv3_' + str(i) + '_a'] = Layer(256, 128, 28, 1, 2)
    else:
        LAYERS['conv3_' + str(i) + '_a'] = Layer(512, 128, 28, 1)
    LAYERS['conv3_' + str(i) + '_b'] = Layer(128, 128, 28, 3)
    LAYERS['conv3_' + str(i) + '_c'] = Layer(128, 512, 28, 1)

for i in range(1, 37):
    if i == 1:
        LAYERS['conv4_' + str(i) + '_a'] = Layer(512, 256, 14, 1, 2)
    else:
        LAYERS['conv4_' + str(i) + '_a'] = Layer(1024, 256, 14, 1)
    LAYERS['conv4_' + str(i) + '_b'] = Layer(256, 256, 14, 3)
    LAYERS['conv4_' + str(i) + '_c'] = Layer(256, 1024, 14, 1)

for i in range(1, 4):
    if i == 1:
        LAYERS['conv5_' + str(i) + '_a'] = Layer(1024, 512, 7, 1, 2)
    else:
        LAYERS['conv5_' + str(i) + '_a'] = Layer(2048, 512, 7, 1)
    LAYERS['conv5_' + str(i) + '_b'] = Layer(512, 512, 7, 3)
    LAYERS['conv5_' + str(i) + '_c'] = Layer(512, 2048, 7, 1)

