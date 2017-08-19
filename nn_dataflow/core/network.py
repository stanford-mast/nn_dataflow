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

from .layer import Layer, InputLayer

class Network(object):
    '''
    NN topology. Support DAG structure of layers.
    '''

    INPUT_LAYER_KEY = '__INPUT__'

    def __init__(self, net_name):
        self.net_name = net_name
        self.layer_dict = OrderedDict()
        self.prevs_dict = {}
        self.nexts_dict = {}

        self._merge_symbol_cache = {}

    def set_input(self, input_layer):
        '''
        Set the input layer.
        '''
        if self.INPUT_LAYER_KEY in self.layer_dict:
            raise KeyError('Network: only one input layer is allowed.')

        if not isinstance(input_layer, InputLayer):
            raise TypeError('Network: input_layer must be an InputLayer '
                            'instance.')

        self.layer_dict[self.INPUT_LAYER_KEY] = input_layer

    def add(self, layer_name, layer, prevs=None):
        '''
        Add a named layer, with optional previous layer(s).

        If previous layer(s) is not given, assume it follows the last added
        layer.
        '''
        if self.INPUT_LAYER_KEY not in self.layer_dict:
            raise RuntimeError('Network: must first set input layer.')

        if layer_name in self.layer_dict:
            raise KeyError('Network: layer {} already exists.'
                           .format(layer_name))

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
                    raise KeyError('Network: given previous layer {} '
                                   'has not been added to the network'.
                                   format(pl))
        else:
            prevs = (self.layer_dict.keys()[-1],)

        self.layer_dict[layer_name] = layer
        self.prevs_dict[layer_name] = prevs

        # Ensure dimension matching between layers.
        try:
            self._merge_symbol(layer_name)
        except ValueError:
            del self.layer_dict[layer_name]
            del self.prevs_dict[layer_name]
            raise

        for pl in prevs:
            self.nexts_dict.setdefault(pl, []).append(layer_name)

    def prev_layers(self, layer_name):
        '''
        Get the previous layers of the given layer name, and the merge approach.

        Return a tuple of all the previous layer names, and the merge symbol.
        Use `None` to represent the input layer in the returned tuple.
        '''
        if layer_name == self.INPUT_LAYER_KEY:
            raise ValueError('Network: cannot get previous layers for '
                             'input layer.')

        prevs = tuple(None if pl == self.INPUT_LAYER_KEY else pl
                      for pl in self.prevs_dict[layer_name])
        assert prevs
        merge_symbol = self._merge_symbol(layer_name)

        return prevs, merge_symbol

    def next_layers(self, layer_name):
        '''
        Get the next layers of the given layer name, i.e., the layers that need
        the output of this layer.

        Return a tuple of all the next layer names. Use `None` to represent the
        output of the last layer in the returned tuple.
        '''
        try:
            nexts = tuple(self.nexts_dict[layer_name])
        except KeyError:
            nexts = tuple([None])
        assert nexts

        return nexts

    def input_layer(self):
        '''
        Get the input layer.
        '''
        return self.layer_dict[self.INPUT_LAYER_KEY]

    def first_layers(self):
        '''
        Get a tuple of the first layers, i.e., those with only the input layer
        as their previous layers.

        If a layer has other layers and the input layer as its previous layers,
        it does not count as a first layer.
        '''
        first_layers = []
        for layer_name in self:
            prevs, _ = self.prev_layers(layer_name)
            if len(prevs) == 1 and None in prevs:
                first_layers.append(layer_name)
        return tuple(first_layers)

    def last_layers(self):
        '''
        Get a tuple of the last layers, i.e., those with no next layer.
        '''
        last_layers = []
        for layer_name in self:
            nexts = self.next_layers(layer_name)
            if len(nexts) == 1 and None in nexts:
                last_layers.append(layer_name)
        return tuple(last_layers)

    def _merge_symbol(self, layer_name):
        '''
        Get the symbol to merge the previous layers as the input to the given
        layer.
        '''
        if layer_name in self._merge_symbol_cache:
            return self._merge_symbol_cache[layer_name]

        layer = self.layer_dict[layer_name]

        prev_layer_names = self.prevs_dict[layer_name]
        assert prev_layer_names

        # Compare the ifmap dimensions of this layer, with all the ofmaps of
        # the previous layers.
        sum_nfmaps = 0
        same_nfmaps = True

        for pln in prev_layer_names:
            pl = self.layer_dict[pln]

            # Ensure fmap sizes match. Allow padding.
            if not layer.is_valid_padding_sifm((pl.hofm, pl.wofm)):
                raise ValueError('Network: {}, a previous layer of {}, '
                                 'has mismatch fmap size: {} vs. {}.'
                                 .format(pln, layer_name,
                                         (pl.hofm, pl.wofm),
                                         (layer.hofm, layer.wofm)))

            sum_nfmaps += pl.nofm
            same_nfmaps = same_nfmaps and pl.nofm == layer.nifm

        if sum_nfmaps == layer.nifm:
            if same_nfmaps:
                assert len(prev_layer_names) == 1
            # Fmaps are concatenated.
            symbol = '|'
        elif same_nfmaps:
            # Fmaps are summed up.
            symbol = '+'
        else:
            raise ValueError('Network: cannot figure out how to merge {}, '
                             'which are the previous layers of {}'
                             .format(' '.join(prev_layer_names), layer_name))

        self._merge_symbol_cache[layer_name] = symbol
        return symbol

    def __contains__(self, layer_name):
        ''' Whether the network contains a layer. '''
        return layer_name in self.layer_dict

    def __len__(self):
        ''' Number of layers in the network. '''
        if self.INPUT_LAYER_KEY not in self.layer_dict:
            assert not self.layer_dict
            return 0
        return len(self.layer_dict) - 1

    def __iter__(self):
        ''' Iterate through layer names. '''
        for layer_name in self.layer_dict.keys():
            if layer_name == self.INPUT_LAYER_KEY:
                continue
            yield layer_name

    def __getitem__(self, layer_name):
        ''' Get the layer by name. '''
        try:
            return self.layer_dict[layer_name]
        except KeyError as e:
            raise KeyError('Network: {} layer not found.'.format(str(e)))

    def __str__(self):
        str_ = 'Network: {}\n'.format(self.net_name)
        for layer_name in self:
            prev_layer_names, merge_symbol = self.prev_layers(layer_name)
            prev_str = merge_symbol.join(['None' if n is None else n
                                          for n in prev_layer_names])
            str_ += '  Layer {} <- {}\n'.format(
                layer_name, ' {} '.format(prev_str))
        return str_

