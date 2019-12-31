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
from nn_dataflow.core import InputLayer, FCLayer

from nn_dataflow.nns import add_lstm_cell

'''
LSTM for phoneme classification.

Graves and Schmidhuber, 2005
'''

NN = Network('PHONEME')

NN.set_input_layer(InputLayer(26, 1))

# Input.
NN.add('We', FCLayer(26, 140), prevs=(NN.INPUT_LAYER_KEY,))

# LSTM.
C, H = add_lstm_cell(NN, 'cell', 140, 'We')

# Output.
NN.add('Wd', FCLayer(140, 61), prevs=(H,))

