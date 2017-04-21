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

class Layer(object):
    '''
    NN layer parameters.

    nifm (C): # ifmap channels
    nofm (M): # ofmap channels
    hifm, wifm (H): ifmap height/width
    hofm, wofm (E): ofmap height/width
    sfil (R): weight filter width/height
    strd (U): stride size
    '''
    # pylint: disable=too-many-instance-attributes

    def __init__(self, nifm, nofm, sofm, sfil, strd=1):
        self.nifm = nifm
        self.nofm = nofm
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
        self.hofm = hofm
        self.hifm = sfil + (hofm - 1) * strd
        self.wofm = wofm
        self.wifm = sfil + (wofm - 1) * strd
        self.sfil = sfil
        self.strd = strd
        assert self.hofm > 0 and self.wofm > 0

    def ifmap_size(self, word_size=1):
        '''
        Get size of one input fmap.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.hifm * self.wifm * word_size

    def total_ifmap_size(self, word_size=1):
        '''
        Get total size of all input fmaps.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.nifm * self.ifmap_size(word_size)

    def ofmap_size(self, word_size=1):
        '''
        Get size of one output fmap.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.hofm * self.wofm * word_size

    def total_ofmap_size(self, word_size=1):
        '''
        Get total size of all output fmaps.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.nofm * self.ofmap_size(word_size)

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

    def total_ops(self, batch_size=1):
        '''
        Get total number of MAC ops.
        '''
        return self.hofm * self.wofm * self.sfil * self.sfil \
                * self.nofm * self.nifm * batch_size


class FCLayer(Layer):
    '''
    NN fully-connected layer parameters.

    hifm = wifm = sfil, strd = 1, hofm = wofm = 1
    '''

    def __init__(self, nifm, nofm, sfil):
        Layer.__init__(self, nifm, nofm, 1, sfil)
        assert self.hofm == 1 and self.wofm == 1

