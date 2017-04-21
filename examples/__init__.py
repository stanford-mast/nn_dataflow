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

def import_network_layers(name):
    '''
    Import an example network.
    '''
    import os
    import importlib

    example_dir = os.path.dirname(os.path.abspath(__file__))
    top_dir = os.path.join(example_dir, '..')
    if not os.path.isfile(os.path.join(example_dir, '__init__.py')):
        raise ImportError
    nets = importlib.import_module('examples.' + name, top_dir)
    layers = nets.LAYERS
    return layers

