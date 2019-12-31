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

from collections import OrderedDict

from .layer import Layer, InputLayer

class Network():
    '''
    NN topology. Support DAG structure of layers.
    '''

    INPUT_LAYER_KEY = '__INPUT__'

    def __init__(self, net_name):
        self.net_name = net_name
        self.layer_dict = OrderedDict()
        self.prevs_dict = {}
        self.nexts_dict = {}
        self.ext_dict = OrderedDict()

    def set_input_layer(self, input_layer):
        '''
        Set the input layer.
        '''
        if self.INPUT_LAYER_KEY in self.layer_dict:
            raise KeyError('Network: only one input layer is allowed.')

        if not isinstance(input_layer, InputLayer):
            raise TypeError('Network: input_layer must be an InputLayer '
                            'instance.')

        self.layer_dict[self.INPUT_LAYER_KEY] = input_layer

    def input_layer(self):
        '''
        Get the input layer.
        '''
        return self.layer_dict[self.INPUT_LAYER_KEY]

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
            for p in prevs:
                try:
                    self.__getitem__(p)
                except KeyError:
                    raise KeyError('Network: given previous layer {} '
                                   'has not been added to the network'.
                                   format(p))
        else:
            prevs = (list(self.layer_dict.keys())[-1],)

        self.layer_dict[layer_name] = layer
        self.prevs_dict[layer_name] = prevs

        # Ensure dimension matching between layers.
        try:
            self._check_prevs(layer_name)
        except ValueError:
            del self.layer_dict[layer_name]
            del self.prevs_dict[layer_name]
            raise

        for p in prevs:
            self.nexts_dict.setdefault(p, []).append(layer_name)

    def add_ext(self, layer_name, layer):
        '''
        Add a named external layer.
        '''
        if layer_name in self.ext_dict:
            raise KeyError('Network: external layer {} already exists.'
                           .format(layer_name))

        if not isinstance(layer, InputLayer):
            raise TypeError('Network: external layer must be an InputLayer '
                            'instance.')

        self.ext_dict[layer_name] = layer

    def prevs(self, layer_name):
        '''
        Get the previous layers of the given layer name.

        Return a tuple of all the previous layer names. Use `None` to represent
        the input layer in the returned tuple.
        '''
        if layer_name == self.INPUT_LAYER_KEY:
            raise ValueError('Network: cannot get previous layers for '
                             'input layer.')
        if layer_name in self.ext_dict:
            raise ValueError('Network: cannot get previous layers for '
                             'external layers.')

        prevs = tuple(None if p == self.INPUT_LAYER_KEY else p
                      for p in self.prevs_dict[layer_name])
        assert prevs

        return prevs

    def nexts(self, layer_name):
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

    def firsts(self):
        '''
        Get a tuple of the first layers, i.e., those with only the input layer
        or external layers as their previous layers.

        If a layer has other layers besides the input/external layers as its
        previous layers, it does not count as a first layer.
        '''
        input_ext_layers = set([None]).union(self.ext_layers())
        firsts = []
        for layer_name in self:
            prevs = self.prevs(layer_name)
            if input_ext_layers.issuperset(prevs):
                firsts.append(layer_name)
        return tuple(firsts)

    def lasts(self):
        '''
        Get a tuple of the last layers, i.e., those with no next layer.
        '''
        lasts = []
        for layer_name in self:
            nexts = self.nexts(layer_name)
            if nexts == (None,):
                lasts.append(layer_name)
        return tuple(lasts)

    def ext_layers(self):
        '''
        Get a tuple of the external layers.
        '''
        return tuple(self.ext_dict.keys())

    def _check_prevs(self, layer_name):
        '''
        Check the previous layers of the given layer name.
        '''
        layer = self.layer_dict[layer_name]

        prevs = self.prevs_dict[layer_name]
        assert prevs

        # Compare the ifmap dimensions of this layer, with all the ofmaps of
        # the previous layers.
        sum_nfmaps = 0

        for p in prevs:
            pl = self.__getitem__(p)

            # Ensure fmap sizes match. Allow padding.
            if not layer.is_valid_padding_sifm((pl.hofm, pl.wofm)):
                raise ValueError('Network: {}, a previous layer of {}, '
                                 'has mismatch fmap size: {} vs. {}.'
                                 .format(p, layer_name,
                                         (pl.hofm, pl.wofm),
                                         (layer.hofm, layer.wofm)))

            sum_nfmaps += pl.nofm

        if sum_nfmaps != layer.nifm:
            raise ValueError('Network: {} cannot be the previous layers of {}.'
                             .format(' | '.join(prevs), layer_name))

    def __contains__(self, layer_name):
        ''' Whether the network contains a layer. '''
        return layer_name in self.layer_dict or layer_name in self.ext_dict

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
        except KeyError:
            try:
                return self.ext_dict[layer_name]
            except KeyError as e:
                raise KeyError('Network: {} layer not found.'.format(str(e)))

    def __str__(self):
        str_ = 'Network: {}\n'.format(self.net_name)
        for layer_name in self:
            prevs = self.prevs(layer_name)
            prev_str = ' | '.join(['None' if n is None else n for n in prevs])
            str_ += '  Layer {} <- {}\n'.format(layer_name, prev_str)
        return str_

