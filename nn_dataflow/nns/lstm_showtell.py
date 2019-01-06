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

from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, EltwiseLayer

from nn_dataflow.nns import add_lstm_cell

'''
LSTM from Show and Tell.

Vinyals et al., Google, CVPR 2015
'''

NN = Network('ShowTell')

NN.set_input_layer(InputLayer(512, 1))

C = H = None

# Unroll by the sequence length, assuming 10.
for idx in range(10):

    # Word embedding is a simple lookup.
    we = 'We_{}'.format(idx)
    NN.add(we, EltwiseLayer(512, 1, 1), prevs=(NN.INPUT_LAYER_KEY,))

    # LSTM.
    cell = 'cell_{}'.format(idx)
    C, H = add_lstm_cell(NN, cell, 512, we, C, H)

    # log(p), softmax.
    wd = 'Wd_{}'.format(idx)
    NN.add(wd, EltwiseLayer(512, 1, 1), prevs=(H,))

