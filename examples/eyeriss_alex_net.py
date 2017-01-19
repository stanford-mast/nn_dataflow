'''
AlexNet

Used in Eyeriss paper.
'''

from collections import OrderedDict

from nn_dataflow import Layer, FCLayer

LAYERS = OrderedDict()

LAYERS['conv1'] = Layer(3, 96, 55, 11, 4)
LAYERS['conv2'] = Layer(48, 256, 27, 5)
LAYERS['conv3'] = Layer(256, 384, 13, 3)
LAYERS['conv4'] = Layer(192, 384, 13, 3)
LAYERS['conv5'] = Layer(192, 256, 13, 3)
LAYERS['fc1'] = FCLayer(256, 4096, 6)
LAYERS['fc2'] = FCLayer(4096, 4096, 1)
LAYERS['fc3'] = FCLayer(4096, 1000, 1)

