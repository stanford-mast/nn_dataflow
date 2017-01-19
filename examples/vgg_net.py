'''
VGGNet-16

Simonyan and Zisserman, 2014
'''

from collections import OrderedDict

from nn_dataflow import Layer, FCLayer

LAYERS = OrderedDict()

LAYERS['conv1'] = Layer(3, 64, 224, 3)
LAYERS['conv2'] = Layer(64, 64, 224, 3)

LAYERS['conv3'] = Layer(64, 128, 112, 3)
LAYERS['conv4'] = Layer(128, 128, 112, 3)

LAYERS['conv5'] = Layer(128, 256, 56, 3)
LAYERS['conv6'] = Layer(256, 256, 56, 3)
LAYERS['conv7'] = Layer(256, 256, 56, 3)

LAYERS['conv8'] = Layer(256, 512, 28, 3)
LAYERS['conv9'] = Layer(512, 512, 28, 3)
LAYERS['conv10'] = Layer(512, 512, 28, 3)

LAYERS['conv11'] = Layer(512, 512, 14, 3)
LAYERS['conv12'] = Layer(512, 512, 14, 3)
LAYERS['conv13'] = Layer(512, 512, 14, 3)

LAYERS['fc1'] = FCLayer(512, 4096, 7)
LAYERS['fc2'] = FCLayer(4096, 4096, 1)
LAYERS['fc3'] = FCLayer(4096, 1000, 1)

