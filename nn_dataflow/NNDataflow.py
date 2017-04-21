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

from . import ParallelEnum as pe
from .PartitionScheme import PartitionScheme
from .PhyDim2 import PhyDim2
from .Scheduling import SchedulingCondition, Scheduling

'''
Top-level scheduling interface.
'''

def schedule_search(layers, batch_size, resource, cost, map_strategy_class,
                    options):
    '''
    Search the best schedule results for the given network and batch size.
    '''

    aggr_tops = [(0, OrderedDict()) for _ in range(options.ntops)]

    # Assume the first layer input is fully fmap partitioned (image tiled).
    partition2d_all_ofmp = [PhyDim2(1, 1) for _ in range(pe.NUM)]
    partition2d_all_ofmp[pe.OFMP] = resource.dim_nodes

    # Keep all previous layer partition schemes appeared in the top schedules.
    # Explore all of them for next layer.
    part_lprev_list = [PartitionScheme(range(pe.NUM), partition2d_all_ofmp)]
    # The corresponding indexes of schedules in aggr_tops for the previous layer
    # partition scheme.
    aggr_top_indexes_list = [range(options.ntops)]

    for name, layer in layers.items():

        layer_sched = Scheduling(layer, batch_size, cost, map_strategy_class)

        new_aggr_tops = []

        # For each previous layer partition scheme, search top schedules for
        # the current layer.
        for part_lprev, aggr_top_indexes in zip(part_lprev_list,
                                                aggr_top_indexes_list):

            condition = SchedulingCondition(resource=resource,
                                            part_src=part_lprev)

            try:
                tops = layer_sched.schedule_search(condition, options)
            except Exception:
                sys.stderr.write('Failed when scheduling layer {}\n'
                                 .format(name))
                raise

            # Append all the current layer top schedules to all the previous top
            # schedules with the matching partition scheme.
            for t_idx in range(options.ntops):
                if t_idx >= len(tops):
                    break
                assert tops[t_idx].dict_part['part_src'] == part_lprev.__dict__
                for at_idx in aggr_top_indexes:
                    new_schedule = aggr_tops[at_idx][1].copy()
                    new_schedule.update({name: tops[t_idx]})
                    atop = (aggr_tops[at_idx][0] + tops[t_idx].total_cost,
                            new_schedule)
                    new_aggr_tops.append(atop)

        # Always pick and keep top n at each layer.
        aggr_tops = sorted(new_aggr_tops, key=lambda x: x[0])[:options.ntops]

        # Record all layer partition schemes for next layer.
        part_lprev_list = []
        aggr_top_indexes_list = []
        for at_idx in range(options.ntops):
            if at_idx >= len(aggr_tops):
                break
            # 1: list of schedules for layers; name: last layer; 2: dict_part.
            # Translate back to Partition2dScheme.
            part_lprev_dict = aggr_tops[at_idx][1][name].dict_part['part']
            part_lprev = PartitionScheme(part_lprev_dict['order'],
                                         part_lprev_dict['pdims'])
            try:
                i = part_lprev_list.index(part_lprev)
            except ValueError:
                assert part_lprev_list.count(part_lprev) == 0
                part_lprev_list.append(part_lprev)
                aggr_top_indexes_list.append([])
                assert len(part_lprev_list) == len(aggr_top_indexes_list)
                i = -1
            aggr_top_indexes_list[i].append(at_idx)

    return aggr_tops

