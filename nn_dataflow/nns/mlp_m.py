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

'''
MLP-M

PRIME, 2016
'''

NN = Network('MLP-M')

NN.set_input_layer(InputLayer(784, 1))

NN.add('fc1', FCLayer(784, 1000))
NN.add('fc2', FCLayer(1000, 500))
NN.add('fc3', FCLayer(500, 250))
NN.add('fc4', FCLayer(250, 10))

