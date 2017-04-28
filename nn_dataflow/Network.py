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

from collections import OrderedDict

from .Layer import Layer

class Network(object):
    '''
    NN topology. Support DAG structure of layers.
    '''

    def __init__(self, net_name):
        self.net_name = net_name
        self.layer_dict = OrderedDict()
        self.prevs_dict = {}

    def add(self, layer_name, layer, prevs=None):
        '''
        Add a named layer, with optional previous layer(s).

        If previous layer(s) is not given, assume it follows the last added
        layer.
        '''
        if not isinstance(layer, Layer):
            raise TypeError('Network: layer must be a Layer instance.')

        # First figure out previous layers.
        if prevs:
            # Ensure `prevs` as a tuple.
            if isinstance(prevs, str):
                prevs = (prevs,)
            else:
                prevs = tuple(prevs)
            # Ensure previous layers are already added.
            for pl in prevs:
                if pl not in self.layer_dict:
                    raise ValueError('Network: given previous layer {} '
                                     'has not been added to the network'.
                                     format(pl))
        elif self.layer_dict:
            # Has previously added layers.
            prevs = (self.layer_dict.keys()[-1],)
        else:
            prevs = tuple()

        self.layer_dict[layer_name] = layer
        self.prevs_dict[layer_name] = prevs

    def layers(self):
        '''
        Generator to iterate through layers. Return layer name and layer.
        '''
        for layer_name, layer in self.layer_dict.items():
            yield layer_name, layer

    def prev_layers(self, layer_name):
        '''
        Get the previous layers of the given layer name, and the merge approach.

        Return a tuple of all the previous layer names, and the merge symbol.
        '''
        return self.prevs_dict[layer_name], self._merge_symbol(layer_name)

    def _merge_symbol(self, layer_name):
        '''
        Get the symbol to merge the previous layers as the input to the given
        layer.
        '''
        layer = self.layer_dict[layer_name]

        prev_layer_names = self.prevs_dict[layer_name]

        if len(prev_layer_names) == 1:
            return ''

        sum_nfmaps = 0
        same_nfmaps = True
        for pln in prev_layer_names:
            pl = self.layer_dict[pln]
            if pl.ofmap_size() != layer.ofmap_size() * layer.strd * layer.strd \
                    and pl.ofmap_size() != layer.ifmap_size():
                # With or without padding.
                raise ValueError('Network: {}, a previous layer of {}, '
                                 'has mismatch fmap size.'
                                 .format(pln, layer_name))
            sum_nfmaps += pl.nofm
            if pl.nofm != layer.nifm:
                same_nfmaps = False

        if sum_nfmaps == layer.nifm:
            # Fmaps are concatenated.
            assert not same_nfmaps
            return '|'
        elif same_nfmaps:
            # Fmaps are summed up.
            return '+'
        else:
            raise ValueError('Network: cannot figure out how to merge {}, '
                             'which are the previous layers of {}'
                             .format(' '.join(prev_layer_names), layer_name))

    def __contains__(self, layer_name):
        ''' Whether the network contains a layer. '''
        return layer_name in self.layer_dict

    def __str__(self):
        str_ = 'Network: {}\n'.format(self.net_name)
        for layer_name in self.layer_dict.keys():
            prev_layer_names, merge_symbol = self.prev_layers(layer_name)
            str_ += '  Layer {} <- {}\n'.format(
                layer_name, ' {} '.format(merge_symbol).join(prev_layer_names))
        return str_

