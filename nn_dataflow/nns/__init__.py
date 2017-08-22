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

