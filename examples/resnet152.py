'''
ResNet-152

He, Zhang, Ren, and Sun, 2015
'''

from collections import OrderedDict

from nn_dataflow import Layer

LAYERS = OrderedDict()

LAYERS['conv1'] = Layer(3, 64, 112, 7, 2)

for i in range(1, 4):
    LAYERS['conv2_' + str(i) + '_a'] = Layer(64, 64, 56, 1)
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

