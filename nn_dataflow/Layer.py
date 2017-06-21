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

from .Util import StringifyClass, ContentHashClass

class Layer(StringifyClass, ContentHashClass):
    '''
    Base NN layer.

    Includes only the output neuron parameters.

    nofm: # ofmap channels
    hofm, wofm: ofmap height/width
    htrd, wtrd: stride height/width
    '''

    def __init__(self, nofm, sofm, strd=1):
        if isinstance(sofm, int):
            hofm = sofm
            wofm = sofm
        elif len(sofm) == 2:
            hofm = sofm[0]
            wofm = sofm[1]
        else:
            raise ValueError('Layer: sofm is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sofm))
        assert hofm > 0 and wofm > 0

        if isinstance(strd, int):
            htrd = strd
            wtrd = strd
        elif len(strd) == 2:
            htrd = strd[0]
            wtrd = strd[1]
        else:
            raise ValueError('Layer: strd is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(strd))
        assert htrd > 0 and wtrd > 0

        self.nofm = nofm
        self.hofm = hofm
        self.wofm = wofm

        self.htrd = htrd
        self.wtrd = wtrd

    def input_layer(self):
        ''' Get the input layer parameters. '''
        raise NotImplementedError(self.__class__.__name__)

    @property
    def nifm(self):
        ''' Number of fmap channels of input layer. '''
        return self.input_layer().nofm

    @property
    def hifm(self):
        ''' Fmap height of input layer. '''
        return self.input_layer().hofm

    @property
    def wifm(self):
        ''' Fmap width of input layer. '''
        return self.input_layer().wofm

    def ofmap_size(self, batch_size=1, word_size=1):
        '''
        Get size of one output fmap with `batch_size`.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.hofm * self.wofm * batch_size * word_size

    def total_ofmap_size(self, batch_size=1, word_size=1):
        '''
        Get total size of all output fmaps with `batch_size`.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.nofm * self.ofmap_size(batch_size, word_size)

    def ifmap_size(self, batch_size=1, word_size=1):
        '''
        Get size of one input fmap with `batch_size`.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.input_layer().ofmap_size(batch_size, word_size)

    def total_ifmap_size(self, batch_size=1, word_size=1):
        '''
        Get total size of all input fmaps with `batch_size`.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.input_layer().total_ofmap_size(batch_size, word_size)

    def ops_per_neuron(self):
        ''' Number of operations per neuron. '''
        raise NotImplementedError(self.__class__.__name__)

    def total_ops(self, batch_size=1):
        ''' Get total number of operations. '''
        return self.total_ofmap_size() * self.ops_per_neuron() * batch_size


class InputLayer(Layer):
    '''
    NN input layer parameters.
    '''

    def input_layer(self):
        return None

    def ops_per_neuron(self):
        return 0


class ConvLayer(Layer):
    '''
    NN convolutional layer parameters.

    nifm (C): # ifmap channels
    nofm (M): # ofmap channels
    hifm, wifm (H): ifmap height/width
    hofm, wofm (E): ofmap height/width
    sfil (R): weight filter width/height
    htrd, wtrd (U): stride height/width
    '''

    def __init__(self, nifm, nofm, sofm, sfil, strd=1):
        super(ConvLayer, self).__init__(nofm, sofm, strd=strd)

        self.sfil = sfil

        hifm = self.sfil + (self.hofm - 1) * self.htrd
        wifm = self.sfil + (self.wofm - 1) * self.wtrd
        self.inlayer = Layer(nifm, (hifm, wifm))

    def input_layer(self):
        return self.inlayer

    def ops_per_neuron(self):
        # 2D convolution across all ifmap channels.
        return self.sfil * self.sfil * self.nifm

    def filter_size(self, word_size=1):
        '''
        Get size of one weight filter.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.sfil * self.sfil * word_size

    def total_filter_size(self, word_size=1):
        '''
        Get total size of all weight filters.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.nifm * self.nofm * self.filter_size(word_size)


class FCLayer(ConvLayer):
    '''
    NN fully-connected layer parameters.

    As a special case of CONVLayer.

    hifm = wifm = sfil, strd = 1, hofm = wofm = 1
    '''

    def __init__(self, nifm, nofm, sfil):
        super(FCLayer, self).__init__(nifm, nofm, 1, sfil)
        assert self.hofm == 1 and self.wofm == 1


class LocalRegionLayer(Layer):
    '''
    NN layer which computes on a local region. The layer has no or limited
    shared weights, whose impact can be ignored during scheduling.

    Includes pooling layer and normalization layer.

    nifm = nofm, sfil = 0
    '''

    def __init__(self, nofm, sofm, nreg, sreg, strd=1):
        super(LocalRegionLayer, self).__init__(nofm, sofm, strd=strd)

        if isinstance(sreg, int):
            hreg = sreg
            wreg = sreg
        elif len(sreg) == 2:
            hreg = sreg[0]
            wreg = sreg[1]
        else:
            raise ValueError('LocalRegionLayer: sreg is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sreg))
        if nreg > 1 and (hreg * wreg) > 1:
            raise ValueError('LocalRegionLayer: local region cannot be a mix '
                             'of both n ({}) and h & w ({}, {})'
                             .format(nreg, hreg, wreg))
        self.nreg = nreg
        self.hreg = hreg
        self.wreg = wreg

        nifm = self.nofm
        hifm = self.hreg + (self.hofm - 1) * self.htrd
        wifm = self.wreg + (self.wofm - 1) * self.wtrd
        self.inlayer = Layer(nifm, (hifm, wifm))

    def input_layer(self):
        return self.inlayer

    def ops_per_neuron(self):
        # Each output point corresponds to merging a local region.
        return self.region_size()

    def region_size(self):
        ''' The size of the local region corresponding to one output point. '''
        return self.nreg * self.hreg * self.wreg


class PoolingLayer(LocalRegionLayer):
    '''
    NN pooling layer parameters.

    As a special case of LocalRegionLayer.

    nreg = 1
    '''

    def __init__(self, nofm, sofm, sreg, strd=None):
        if strd is None:
            strd = sreg
        super(PoolingLayer, self).__init__(nofm, sofm, 1, sreg, strd=strd)
        assert self.nreg == 1

