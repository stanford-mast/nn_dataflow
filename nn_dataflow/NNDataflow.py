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

import sys
from collections import OrderedDict

from . import Partition
from . import Util
from .Cost import Cost
from .DataLayout import DataLayout
from .FmapRange import FmapPosition, FmapRange, FmapRangeMap
from .Layer import Layer
from .Network import Network
from .Resource import Resource
from .Scheduling import SchedulingCondition, SchedulingResult, Scheduling

class SchedulingResultDict(object):
    '''
    Network scheduling result, as a dict of layer scheduling results.

    Include the total cost, and the layer scheduling results as an OrderedDict.
    '''

    def __init__(self, res_dict=None):

        total_cost = 0

        if res_dict is None:
            res_dict = OrderedDict()
        else:
            for name in res_dict:
                res = res_dict[name]
                if isinstance(res, SchedulingResult):
                    total_cost += res.total_cost
                else:
                    raise TypeError('SchedulingResultDict: res_dict value type '
                                    'must be SchedulingResult.')

        self.total_cost = total_cost
        self.res_dict = res_dict

    def __len__(self):
        ''' Get the number of scheduled layers. '''
        return len(self.res_dict)

    def __getitem__(self, layer_name):
        ''' Get the layer SchedulingResult. '''
        return self.res_dict[layer_name]

    def __setitem__(self, layer_name, sched_result):
        ''' In-place update by adding the result of a new layer. '''
        if layer_name in self.res_dict:
            raise KeyError('SchedulingResultDict: layer {} already exists.'
                           .format(layer_name))
        if not isinstance(sched_result, SchedulingResult):
            raise TypeError('SchedulingResultDict: sched_result must be '
                            'a SchedulingResult instance.')
        self.total_cost += sched_result.total_cost
        self.res_dict[layer_name] = sched_result

    def __contains__(self, layer_name):
        ''' Whether the layer is already scheduled. '''
        return layer_name in self.res_dict

    def scheduling_total_cost(self):
        ''' Get the scheduling total cost. '''
        return self.total_cost

    def scheduling_result_dict(self):
        ''' Get the scheduling result dict. '''
        return self.res_dict

    def copy(self):
        ''' Return a shallow copy. '''
        # Shallow copy of layer SchedulingResult is sufficient, since they are
        # read-only.
        return SchedulingResultDict(self.res_dict.copy())

    def __cmp__(self, other):
        if not isinstance(other, SchedulingResultDict):
            raise TypeError('SchedulingResultDict: a SchedulingResultDict '
                            'object is required.')
        if self.total_cost > other.total_cost:
            return 1
        elif self.total_cost < other.total_cost:
            return -1
        return 0


class NNDataflow(object):
    '''
    Search optimized dataflows for neural networks.
    '''
    # pylint: disable=too-few-public-methods

    def __init__(self, network, batch_size, resource, cost):
        if not isinstance(network, Network):
            raise TypeError('NNDataflow: network must be a Network instance.')
        if not isinstance(resource, Resource):
            raise TypeError('NNDataflow: resource must be a Resource instance.')
        if not isinstance(cost, Cost):
            raise TypeError('NNDataflow: cost must be a Cost instance.')

        self.network = network
        self.batch_size = batch_size
        self.resource = resource
        self.cost = cost

    def schedule_search(self, map_strategy_class, options):
        '''
        Search the optimized dataflows.
        '''

        sched_res_dict_list = [SchedulingResultDict()]

        for layer_name in self.network:
            sched_res_dict_list = self._layer_schedule_search(
                layer_name, sched_res_dict_list, map_strategy_class, options)

        return sched_res_dict_list

    def _layer_schedule_search(self, layer_name, sched_res_dict_list,
                               map_strategy_class, options):
        '''
        Schedule the given layer under the previous layer scheduling results.
        `sched_res_dict_list` contains up to top n SchedulingResultDict for the
        previous layers.
        '''

        layer = self.network[layer_name]
        layer_sched = Scheduling(layer, self.batch_size, self.cost,
                                 map_strategy_class)

        new_sched_res_dict_list = []

        for ifmap_layout, srd_idx in self._gen_layer_ifmap_layout(
                layer_name, sched_res_dict_list, options):

            condition = SchedulingCondition(resource=self.resource,
                                            ifmap_layout=ifmap_layout)

            try:
                tops = layer_sched.schedule_search(condition, options)
            except Exception:
                sys.stderr.write('Failed when scheduling layer {}.\n'
                                 .format(layer_name))
                raise

            if not tops:
                sys.stderr.write('Layer {} has no valid schedule.\n'
                                 .format(layer_name))

            # Append all the current layer top schedules to all the previous top
            # schedules with the matching fmap layout.
            for t in tops:
                srd = sched_res_dict_list[srd_idx].copy()
                srd[layer_name] = t
                new_sched_res_dict_list.append(srd)

        # Always pick and keep top n at each layer.
        return sorted(new_sched_res_dict_list)[:options.ntops]

    def _gen_layer_ifmap_layout(self, layer_name, sched_res_dict_list, options):
        '''
        Generator to get all the choices of ifmap layout for the layer.

        Return the ifmap layout, and the corresponding SchedulingResultDict
        index in the list.
        '''

        layer = self.network[layer_name]
        prev_layer_names, merge_symbol = self.network.prev_layers(layer_name)

        if not prev_layer_names:
            # No previous layer, the first layer.
            assert len(sched_res_dict_list) == 1 \
                    and sched_res_dict_list[0].total_cost == 0, \
                    'NNDataflow: initial sched_res_dict_list should only ' \
                    'contain one 0 cost result.'

            for input_layout in self._gen_input_layout(options):
                yield input_layout, 0
            return

        for idx, srd in enumerate(sched_res_dict_list):
            # Merge all previous layer ofmap layouts to get the ifmap layout.
            ifmap_layout = srd[prev_layer_names[0]].ofmap_layout
            for pl_name in prev_layer_names[1:]:
                ifmap_layout = ifmap_layout.merge(merge_symbol,
                                                  srd[pl_name].ofmap_layout)

            # Remap dst memory to src memory.
            origin_diff = self.resource.mem_region_src().origin \
                    - self.resource.mem_region_dst().origin
            ifmap_layout = ifmap_layout.view(origin_diff=origin_diff)

            # Layout dimension check.
            icfrng = ifmap_layout.frmap.complete_fmap_range()
            assert icfrng.size('b') == self.batch_size \
                    and icfrng.size('n') == layer.nifm

            ## FIXME: Hack to deal with fmap size shrink due to pooling.
            icfrng = ifmap_layout.frmap.complete_fmap_range()
            h_shk_flt = 1. * icfrng.size('h') / layer.hifm
            h_shk = int(round(h_shk_flt) + 1e-4)
            if abs(h_shk / h_shk_flt - 1) > 0.3 \
                    or not (h_shk == 1 or h_shk == 2 or h_shk == 4):
                raise ValueError('NNDataflow: fmap shrink by {}?'
                                 .format(h_shk_flt))
            w_shk_flt = 1. * icfrng.size('w') / layer.wifm
            w_shk = int(round(w_shk_flt) + 1e-4)
            if abs(w_shk / w_shk_flt - 1) > 0.3 \
                    or not (w_shk == 1 or w_shk == 2 or w_shk == 4):
                raise ValueError('NNDataflow: fmap shrink by {}?'
                                 .format(w_shk_flt))
            # Make the new layout after shrinking.
            new_frmap = FmapRangeMap()
            for frng, coords in ifmap_layout.frmap.items():
                fpb = frng.fp_beg
                fpe = frng.fp_end
                new_frng = FmapRange(FmapPosition(b=fpb.b, n=fpb.n,
                                                  h=Util.idivc(fpb.h, h_shk),
                                                  w=Util.idivc(fpb.w, w_shk)),
                                     FmapPosition(b=fpe.b, n=fpe.n,
                                                  h=Util.idivc(fpe.h, h_shk),
                                                  w=Util.idivc(fpe.w, w_shk)))
                new_frmap.add(new_frng, coords)

            ifmap_layout = DataLayout(frmap=new_frmap,
                                      origin=ifmap_layout.origin)

            yield ifmap_layout, idx

    def _gen_input_layout(self, options):
        '''
        Get the input layer layout choices.
        '''

        first_layer = self.network[self.network.first_layer_name()]
        input_layer = Layer(nifm=1, nofm=first_layer.nifm,
                            sofm=(first_layer.hifm, first_layer.wifm), sfil=1)

        mem_region = self.resource.mem_region_src()

        for part in Partition.gen_partition(input_layer, self.batch_size,
                                            mem_region.dim, options):
            input_layout = Partition.get_ofmap_layout(
                input_layer, self.batch_size, part, mem_region)

            yield input_layout

