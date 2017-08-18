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

    def __init__(self, network, input_layout):
        # pylint: disable=super-init-not-called

        if not isinstance(network, Network):
            raise TypeError('NNDataflowScheme: network must be a '
                            'Network instance.')
        if not isinstance(input_layout, DataLayout):
            raise TypeError('NNDataflowScheme: input_layout must be a '
                            'DataLayout instance.')
        self.network = network
        self.input_layout = input_layout

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

        prev_layers, _ = self.network.prev_layers(layer_name)
        for pl in prev_layers:
            if pl is None:
                continue
            if pl not in self.res_dict:
                raise KeyError('NNDataflowScheme: layer {} has its previous '
                               'layer {} not scheduled yet.'
                               .format(layer_name, pl))

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
        df = NNDataflowScheme(self.network, self.input_layout)
        for layer_name in self.res_dict:
            df[layer_name] = self.res_dict[layer_name]
        assert util.isclose(df.total_cost, self.total_cost, rel_tol=1e-5)
        assert util.isclose(df.total_time, self.total_time, rel_tol=1e-5)
        return df

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

    @property
    def total_node_time(self):
        ''' Get the total node-time products. '''
        return sum(sr.total_time * sr.dict_part['num_nodes']
                   for sr in self.values())

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
        return 1. * sched_result.total_ops / sched_result.total_time \
                / sched_result.dict_part['num_nodes']

    @staticmethod
    def total_dram_bandwidth(sched_result):
        ''' Layer total DRAM bandwidth in elements per cycle. '''
        return 1. * sched_result.total_accesses[me.DRAM] \
                / sched_result.total_time

