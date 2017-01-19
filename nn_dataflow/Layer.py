'''
NN Layer specification.
'''

class Layer(object):
    '''
    NN layer parameters.

    nifm (C): # ifmap channels
    nofm (M): # ofmap channels
    sifm (H): ifmap width/height
    sofm (E): ofmap width/height
    sfil (R): weight filter width/height
    strd (U): stride size
    '''

    def __init__(self, nifm, nofm, sofm, sfil, strd=1):
        self.nifm = nifm
        self.nofm = nofm
        self.sofm = sofm
        self.sifm = sfil + (sofm - 1) * strd
        self.sfil = sfil
        self.strd = strd
        assert self.sofm > 0

    def ifmap_size(self, word_size=1):
        '''
        Get size of one input fmap.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.sifm * self.sifm * word_size

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
        return self.sofm * self.sofm * word_size

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
        return self.sofm * self.sofm * self.sfil * self.sfil \
                * self.nofm * self.nifm * batch_size


class FCLayer(Layer):
    '''
    NN fully-connected layer parameters.

    sifm = sfil, strd = 1, sofm = 1
    '''

    def __init__(self, nifm, nofm, sfil):
        Layer.__init__(self, nifm, nofm, 1, sfil)
        assert self.sofm == 1


