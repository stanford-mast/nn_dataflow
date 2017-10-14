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

def import_network(name):
    '''
    Import an example network.
    '''
    import importlib

    if name not in all_networks():
        raise ImportError('nns: NN {} has not been defined!'.format(name))
    netmod = importlib.import_module('.' + name, 'nn_dataflow.nns')
    network = netmod.NN
    return network


def all_networks():
    '''
    Get all defined networks.
    '''
    import os

    nns_dir = os.path.dirname(os.path.abspath(__file__))
    nns = [f[:-len('.py')] for f in os.listdir(nns_dir)
           if f.endswith('.py') and not f.startswith('__')]
    return list(sorted(nns))


def add_lstm_cell(network, name, size, xin, cin, hin):
    '''
    Add a LSTM cell named `name` to the `network`, with the dimension `size`.
    `xin`, `cin`, `hin` are the layers' names whose outputs are x_t, C_{t-1},
    h_{t-1}, respectively. Return the layers' names whose outputs are C_t, h_t.
    '''
    from nn_dataflow.core import Network
    from nn_dataflow.core import FCLayer, EltwiseLayer

    if not isinstance(network, Network):
        raise TypeError('add_lstm_cell: network must be a Network instance.')
    if cin not in network or hin not in network or xin not in network:
        raise ValueError('add_lstm_cell: cin {}, hin {}, xin {} must all be '
                         'in the network.'.format(cin, hin, xin))

    def gate_name(gate):
        ''' Name of a gate. '''
        return '{}_{}gate'.format(name, gate)

    # Three gates.
    for g in ['f', 'i', 'o']:
        network.add(gate_name(g), FCLayer(2 * size, size), prevs=(hin, xin))

    # Candidate.
    cand_name = '{}_cand'.format(name)
    network.add(cand_name, FCLayer(2 * size, size), prevs=(hin, xin))

    # C_t.
    cout_name = '{}_cout'.format(name)
    network.add(cout_name, EltwiseLayer(size, 1),
                prevs=(cin, gate_name('f'), cand_name, gate_name('i')))

    # h_t.
    hout_name = '{}_hout'.format(name)
    network.add(hout_name, EltwiseLayer(size, 1),
                prevs=(cout_name, gate_name('o')))

    return cout_name, hout_name

