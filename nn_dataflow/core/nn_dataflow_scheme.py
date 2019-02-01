""" $lic$
Copyright (C) 2016-2019 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

from collections import OrderedDict, MutableMapping

from . import mem_hier_enum as me
from .. import util
from .data_layout import DataLayout
from .network import Network
from .scheduling import SchedulingResult

class NNDataflowScheme(MutableMapping):
    '''
    Neural network dataflow result, as a specialized OrderedDict of layer
    scheduling results.
    '''

    def __init__(self, network, input_layout, ext_layout_dict=None):
        # pylint: disable=super-init-not-called

        if not isinstance(network, Network):
            raise TypeError('NNDataflowScheme: network must be a '
                            'Network instance.')
        if not isinstance(input_layout, DataLayout):
            raise TypeError('NNDataflowScheme: input_layout must be a '
                            'DataLayout instance.')

        if ext_layout_dict is None:
            ext_layout_dict = {}
        if set(ext_layout_dict.keys()) != set(network.ext_layers()):
            raise ValueError('NNDataflowScheme: ext_layout_dict keys do '
                             'not match network.')
        for ext_layout in ext_layout_dict.values():
            if not isinstance(ext_layout, DataLayout):
                raise TypeError('NNDataflowScheme: ext_layout must be a '
                                'DataLayout instance.')

        self.network = network
        self.input_layout = input_layout
        self.ext_layout_dict = ext_layout_dict

        self.res_dict = OrderedDict()

        self.total_cost = 0
        self.total_time = 0

    def __getitem__(self, layer_name):
        ''' Get the SchedulingResult of a scheduled layer. '''
        return self.res_dict[layer_name]

    def __setitem__(self, layer_name, sched_result):
        ''' Add the SchedulingResult of a new layer. '''
        if layer_name not in self.network:
            raise KeyError('NNDataflowScheme: layer {} does not belong to '
                           'network {}.'
                           .format(layer_name, self.network.net_name))
        if layer_name in self.res_dict:
            raise KeyError('NNDataflowScheme: layer {} already exists.'
                           .format(layer_name))
        if not isinstance(sched_result, SchedulingResult):
            raise TypeError('NNDataflowScheme: sched_result must be '
                            'a SchedulingResult instance.')

        prevs = self.network.prevs(layer_name)
        for p in prevs:
            if p is None or p in self.network.ext_layers():
                continue
            if p not in self.res_dict:
                raise KeyError('NNDataflowScheme: layer {} has its previous '
                               'layer {} not scheduled yet.'
                               .format(layer_name, p))

        self.res_dict[layer_name] = sched_result

        self.total_cost += sched_result.total_cost
        self.total_time += sched_result.total_time

    def __delitem__(self, layer_name):
        ''' Not legal to call. '''
        raise KeyError('NNDataflowScheme: cannot delete scheduled layer.')

    def __iter__(self):
        ''' Iterate over scheduled layers. '''
        for layer_name in self.res_dict:
            yield layer_name

    def __len__(self):
        ''' Get the number of scheduled layers. '''
        return len(self.res_dict)

    def copy(self):
        '''
        Return a shallow copy.

        Shallow copy of layer SchedulingResult is sufficient, since they are
        read-only.
        '''
        nndf = NNDataflowScheme(self.network, self.input_layout,
                                self.ext_layout_dict)
        nndf.update(self)
        assert util.isclose(nndf.total_cost, self.total_cost, rel_tol=1e-5)
        assert util.isclose(nndf.total_time, self.total_time, rel_tol=1e-5)
        return nndf

    def fmap_layout(self, layers):
        '''
        Get a DataLayout instance that merges the ofmaps of all given layers.
        '''
        def _ofmap_layout(layer_name):
            if layer_name is None:
                return self.input_layout
            try:
                return self.ext_layout_dict[layer_name]
            except KeyError:
                pass
            return self.res_dict[layer_name].ofmap_layout

        return DataLayout.concat(*[_ofmap_layout(l) for l in layers])

    @property
    def total_ops(self):
        ''' Get the total ops. '''
        return sum(sr.total_ops for sr in self.values())

    @property
    def total_accesses(self):
        ''' Get the total accesses at all memory hierarchies as a list. '''
        accesses = [0] * me.NUM
        for sr in self.values():
            accesses = [a + a1 for a, a1 in zip(accesses, sr.total_accesses)]
        return accesses

    @property
    def total_noc_hops(self):
        ''' Get the total NoC hops. '''
        return sum(sr.total_noc_hops for sr in self.values())

    def perlayer_stats(self, stats_name):
        '''
        Get a dict of per-layer stats. Valid stats must be a static method.
        '''
        try:
            stats_func = getattr(self, stats_name)
        except AttributeError:
            raise AttributeError('NNDataflowScheme: stats {} is not supported.'
                                 .format(stats_name))

        stats = OrderedDict()
        for layer_name in self.res_dict:
            stats[layer_name] = stats_func(self.res_dict[layer_name])

        return stats

    @staticmethod
    def active_node_pes(sched_result):
        ''' Layer active node PE counts. '''
        return 1. * sched_result.total_ops \
                / sched_result.total_proc_time / sched_result.num_nodes

    @staticmethod
    def dram_bandwidth(sched_result):
        ''' Layer total DRAM bandwidth in elements per cycle. '''
        return 1. * sched_result.total_accesses[me.DRAM] \
                / sched_result.total_time

