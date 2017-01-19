'''
ZFNet

Zeiler and Fergus, 2013
'''

from collections import OrderedDict

from nn_dataflow import Layer, FCLayer

LAYERS = OrderedDict()

LAYERS['conv1'] = Layer(3, 96, 110, 7, 2)
LAYERS['conv2'] = Layer(96, 256, 26, 5, 2)
LAYERS['conv3'] = Layer(256, 512, 13, 3)
LAYERS['conv4'] = Layer(512, 1024, 13, 3)
LAYERS['conv5'] = Layer(1024, 512, 13, 3)
LAYERS['fc1'] = FCLayer(512, 4096, 6)
LAYERS['fc2'] = FCLayer(4096, 4096, 1)
LAYERS['fc3'] = FCLayer(4096, 1000, 1)

