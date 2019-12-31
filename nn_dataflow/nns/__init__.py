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


def add_lstm_cell(network, name, size, xin, cin=None, hin=None):
    '''
    Add a LSTM cell named `name` to the `network`, with the dimension `size`.
    `xin`, `cin`, `hin` are the layers' names whose outputs are x_t, C_{t-1},
    h_{t-1}, respectively. Return the layers' names whose outputs are C_t, h_t.
    '''
    from nn_dataflow.core import Network
    from nn_dataflow.core import InputLayer, FCLayer, EltwiseLayer

    if not isinstance(network, Network):
        raise TypeError('add_lstm_cell: network must be a Network instance.')

    if cin is None:
        cin = '{}_cinit'.format(name)
        network.add_ext(cin, InputLayer(size, 1))
    if hin is None:
        hin = '{}_hinit'.format(name)
        network.add_ext(hin, InputLayer(size, 1))

    if (cin not in network) or (hin not in network) or (xin not in network):
        raise ValueError('add_lstm_cell: cin {}, hin {}, xin {} must all be '
                         'in the network.'.format(cin, hin, xin))

    def gate_name(gate):
        ''' Name of a gate. '''
        return '{}_{}gate'.format(name, gate)

    # Candidate.
    cand_name = '{}_cand'.format(name)
    prevs = (hin, xin) if hin else (xin,)
    network.add(cand_name, FCLayer(len(prevs) * size, size), prevs=prevs)

    # Three gates.
    prevs = (hin, xin) if hin else (xin,)
    for g in ['i', 'f', 'o']:
        network.add(gate_name(g), FCLayer(len(prevs) * size, size), prevs=prevs)

    # C_t.
    cout_name = '{}_cout'.format(name)
    cout_f_name = cout_name + '_f'
    prevs = (cin, gate_name('f')) if cin else (gate_name('f'),)
    network.add(cout_f_name, EltwiseLayer(size, 1, len(prevs)), prevs=prevs)
    cout_i_name = cout_name + '_i'
    prevs = (cand_name, gate_name('i'))
    network.add(cout_i_name, EltwiseLayer(size, 1, 2), prevs=prevs)
    prevs = (cout_i_name, cout_f_name)
    network.add(cout_name, EltwiseLayer(size, 1, 2), prevs=prevs)

    # h_t.
    hout_name = '{}_hout'.format(name)
    prevs = (cout_name, gate_name('o'))
    network.add(hout_name, EltwiseLayer(size, 1, 2), prevs=prevs)

    return cout_name, hout_name

