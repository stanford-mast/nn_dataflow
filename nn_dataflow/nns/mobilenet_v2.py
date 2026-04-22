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
MobileNetV2
Sandler, Howard, Zhu, Zhmoginov, and Chen, 2018
https://arxiv.org/abs/1801.04381 
and
https://www.geeksforgeeks.org/computer-vision/what-is-mobilenet-v2/

Input: 224x224x3
Each inverted residual block is expanded as three layers:
  1. 1x1 pointwise expand conv   expand channels by factor t
  2. 3x3 depthwise conv          modeled as ConvLayer with in==out
  3. 1x1 pointwise project conv  project back to output channels

Batch normalization layers are ignored.
'''
# First Layer
# input and initial conv form the first row of table 2 from: https://arxiv.org/pdf/1801.04381
NN = Network('MobileNetV2')
NN.set_input_layer(InputLayer(3, 224))

# Initial conv: t = n/a, c = 32, n = 1, s = 2, 112x112 output
NN.add('conv1', ConvLayer(3, 32, 112, 3, 2), prevs=(NN.INPUT_LAYER_KEY,))

# Next 7 layers: the bottleneck section
# Helpful Notation: 
# t = expansion factor for the num of channels
# c = num of output channels for the block
# n = num of times the block is repeated
# s = the stride for dw conv

# Layer 2
# Block 1: Input = 112, t = 1, c = 16, n = 1, s = 1.
NN.add('b1_dw',  ConvLayer(32, 32, 112, 3),    prevs=('conv1',))
NN.add('b1_pw',  ConvLayer(32, 16, 112, 1),    prevs=('b1_dw',))

# Layer 3
# Block 2.1: Input = 112, t = 6, c = 24, n = 2, s = 2 
NN.add('b2_1_exp', ConvLayer(16, 96, 112, 1),    prevs=('b1_pw',))
NN.add('b2_1_dw',  ConvLayer(96, 96, 56, 3, 2),  prevs=('b2_1_exp',))
NN.add('b2_1_pw',  ConvLayer(96, 24, 56, 1),     prevs=('b2_1_dw',))

# Block 2.2: Input = 56, t = 6, c = 24, n = 2, s = 1
NN.add('b2_2_exp', ConvLayer(24, 144, 56, 1),    prevs=('b2_1_pw',))
NN.add('b2_2_dw',  ConvLayer(144, 144, 56, 3),   prevs=('b2_2_exp',))
NN.add('b2_2_pw',  ConvLayer(144, 24, 56, 1),    prevs=('b2_2_dw',))

# Layer 4
# Block 3.1: Input = 56, t = 6, c = 32, n = 3, s = 2 
NN.add('b3_1_exp', ConvLayer(24, 144, 56, 1),    prevs=('b2_2_pw',))
NN.add('b3_1_dw',  ConvLayer(144, 144, 28, 3, 2),prevs=('b3_1_exp',))
NN.add('b3_1_pw',  ConvLayer(144, 32, 28, 1),    prevs=('b3_1_dw',))

# Block 3.2: input = 28 t = 6, c = 32, n = 3, s = 1 
NN.add('b3_2_exp', ConvLayer(32, 192, 28, 1),    prevs=('b3_1_pw',))
NN.add('b3_2_dw',  ConvLayer(192, 192, 28, 3),   prevs=('b3_2_exp',))
NN.add('b3_2_pw',  ConvLayer(192, 32, 28, 1),    prevs=('b3_2_dw',))

# Block 3.3: input = 28 t = 6, c = 32, n = 3, s = 1
NN.add('b3_3_exp', ConvLayer(32, 192, 28, 1),    prevs=('b3_2_pw',))
NN.add('b3_3_dw',  ConvLayer(192, 192, 28, 3),   prevs=('b3_3_exp',))
NN.add('b3_3_pw',  ConvLayer(192, 32, 28, 1),    prevs=('b3_3_dw',))

# Layer 5
# Block 4.1: Input = 28, t = 6, c = 64, n = 4, s = 2
NN.add('b4_1_exp', ConvLayer(32, 192, 28, 1),    prevs=('b3_3_pw',))
NN.add('b4_1_dw',  ConvLayer(192, 192, 14, 3, 2),prevs=('b4_1_exp',))
NN.add('b4_1_pw',  ConvLayer(192, 64, 14, 1),    prevs=('b4_1_dw',))

# Block 4.2: Input = 14, t = 6, c = 64, n = 4, s = 1
NN.add('b4_2_exp', ConvLayer(64, 384, 14, 1),    prevs=('b4_1_pw',))
NN.add('b4_2_dw',  ConvLayer(384, 384, 14, 3),   prevs=('b4_2_exp',))
NN.add('b4_2_pw',  ConvLayer(384, 64, 14, 1),    prevs=('b4_2_dw',))

# Block 4.3: Input = 14, t = 6, c = 64, n = 4, s = 1
NN.add('b4_3_exp', ConvLayer(64, 384, 14, 1),    prevs=('b4_2_pw',))
NN.add('b4_3_dw',  ConvLayer(384, 384, 14, 3),   prevs=('b4_3_exp',))
NN.add('b4_3_pw',  ConvLayer(384, 64, 14, 1),    prevs=('b4_3_dw',))

# Block 4.4: Input = 14, t = 6, c = 64, n = 4, s = 1
NN.add('b4_4_exp', ConvLayer(64, 384, 14, 1),   prevs=('b4_3_pw',))
NN.add('b4_4_dw',  ConvLayer(384, 384, 14, 3),  prevs=('b4_4_exp',))
NN.add('b4_4_pw',  ConvLayer(384, 64, 14, 1),   prevs=('b4_4_dw',))

# Layer 6
# Block 5.1: Input = 14, t=6, c=96, n = 3, s = 1
NN.add('b5_1_exp', ConvLayer(64, 384, 14, 1),   prevs=('b4_4_pw',))
NN.add('b5_1_dw',  ConvLayer(384, 384, 14, 3),  prevs=('b5_1_exp',))
NN.add('b5_1_pw',  ConvLayer(384, 96, 14, 1),   prevs=('b5_1_dw',))

# Block 5.2: Input = 14, t=6, c=96, n = 3, s = 1
NN.add('b5_2_exp', ConvLayer(96, 576, 14, 1),   prevs=('b5_1_pw',))
NN.add('b5_2_dw',  ConvLayer(576, 576, 14, 3),  prevs=('b5_2_exp',))
NN.add('b5_2_pw',  ConvLayer(576, 96, 14, 1),   prevs=('b5_2_dw',))

# Block 5.3: Input = 14, t=6, c=96, n = 3, s = 1
NN.add('b5_3_exp', ConvLayer(96, 576, 14, 1),   prevs=('b5_2_pw',))
NN.add('b5_3_dw',  ConvLayer(576, 576, 14, 3),  prevs=('b5_3_exp',))
NN.add('b5_3_pw',  ConvLayer(576, 96, 14, 1),   prevs=('b5_3_dw',))

# Layer 7
# Block 6.1: Input = 14, t = 6, c = 160, n = 3, s = 2
NN.add('b6_1_exp', ConvLayer(96, 576, 14, 1),   prevs=('b5_3_pw',))
NN.add('b6_1_dw',  ConvLayer(576, 576, 7, 3, 2),prevs=('b6_1_exp',))
NN.add('b6_1_pw',  ConvLayer(576, 160, 7, 1),   prevs=('b6_1_dw',))

# Block 6.2: Input = 7, t = 6, c = 160, n = 3, s = 1
NN.add('b6_2_exp', ConvLayer(160, 960, 7, 1),   prevs=('b6_1_pw',))
NN.add('b6_2_dw',  ConvLayer(960, 960, 7, 3),   prevs=('b6_2_exp',))
NN.add('b6_2_pw',  ConvLayer(960, 160, 7, 1),   prevs=('b6_2_dw',))

# Block 6.3: Input = 7, t = 6, c = 160, n = 3, s = 1
NN.add('b6_3_exp', ConvLayer(160, 960, 7, 1),   prevs=('b6_2_pw',))
NN.add('b6_3_dw',  ConvLayer(960, 960, 7, 3),   prevs=('b6_3_exp',))
NN.add('b6_3_pw',  ConvLayer(960, 160, 7, 1),   prevs=('b6_3_dw',))

# Layer 8: last bottleneck block
# Block 7: Input = 7, t = 6, c = 320, n = 1, s = 1
NN.add('b7_exp', ConvLayer(160, 960, 7, 1),   prevs=('b6_3_pw',))
NN.add('b7_dw',  ConvLayer(960, 960, 7, 3),   prevs=('b7_exp',))
NN.add('b7_pw',  ConvLayer(960, 320, 7, 1),   prevs=('b7_dw',))

# Last pointwise conv: Input = 7, t = n/a, c = 1280, n = 1, s = 1
NN.add('conv_last', ConvLayer(320, 1280, 7, 1), prevs=('b7_pw',))

# Global average pool: n = 1
NN.add('pool_last', PoolingLayer(1280, 1, 7, strd=7), prevs=('conv_last',))

# Classifier: c = k
NN.add('fc', FCLayer(1280, 1000, 1), prevs=('pool_last',))