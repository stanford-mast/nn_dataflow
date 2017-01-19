'''
Neural network examples.
'''


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

