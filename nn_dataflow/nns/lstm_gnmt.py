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
from nn_dataflow.core import InputLayer, FCLayer

from nn_dataflow.nns import add_lstm_cell

'''
LSTM from GNMT.

Sutskever, Vinyals, Le, Google, NIPS 2014
'''

NN = Network('GNMT')

NN.set_input_layer(InputLayer(80000, 1))

NN.add('init', FCLayer(80000, 1000))

NL = 4
CL = ['init'] * NL
HL = ['init'] * NL

# Unroll by the sequence length, assuming 10.
for idx in range(10):

    new_CL = []
    new_HL = []

    we = 'We_{}'.format(idx)
    NN.add(we, FCLayer(80000, 1000), prevs=(NN.INPUT_LAYER_KEY,))
    x = we

    for l in range(NL):
        cell = 'cell_l{}_{}'.format(l, idx)
        C, H = add_lstm_cell(NN, cell, 1000, x, CL[l], HL[l])
        new_CL.append(C)
        new_HL.append(H)
        x = H

    wd = 'Wd_{}'.format(idx)
    NN.add(wd, FCLayer(1000, 80000), prevs=(x,))

    CL = new_CL
    HL = new_HL

