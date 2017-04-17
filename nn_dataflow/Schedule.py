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

'''
Top-level scheduling interface.
'''

import heapq
import math
import multiprocessing
import sys
from collections import OrderedDict

from . import DataCategoryEnum as de
from . import LoopBlocking
from . import MemHierEnum as me
from . import ParallelEnum as pe
from . import Partition
from . import Solver
from .Partition import Partition2dScheme
from .PhyDim2 import PhyDim2


def _get_loopblocking_genfunc(options):
    ''' Get the generator function for loop blocking. '''
    if options.sw_solve_loopblocking:
        return Solver.gen_loopblocking_gbuf_regf
    else:
        return LoopBlocking.gen_loopblocking_gbuf_regf


def _get_partition2d_genfunc(options):
    ''' Get the generator function for parallel partition. '''
    if options.partition_hybrid:
        return Partition.gen_layer_partition2d
    else:
        return Partition.gen_layer_naive_partition2d


def _combine_search_lpbl_part2d(resource, cost, nested_loop_desc, layer_part,
                                batch_size, part_lcurr, part_lprev, unit_nhops,
                                options):
    '''
    Combine search the best loop blocking schemes with a certain parallel
    partition scheme with the smallest combined cost.

    `nested_loop_desc` and `layer_part` are for the layer after partitioning.

    `unit_nhops` is the unit number of hops for each data category for a single
    access run.
    '''

    layer_data_size = [0] * de.NUM
    layer_data_size[de.FIL] = layer_part.total_filter_size()
    layer_data_size[de.IFM] = layer_part.total_ifmap_size() * batch_size
    layer_data_size[de.OFM] = layer_part.total_ofmap_size() * batch_size

    def sweep(): # pylint: disable=missing-docstring

        for cost_loop, dict_loop in _get_loopblocking_genfunc(options)(
                resource, cost, nested_loop_desc, options):

            if math.isinf(cost_loop):
                continue

            # Calculate total number of hops, which determined by both
            # partition scheme (number of hops per access) and loop blocking
            # (runs of access).
            access_mem = dict_loop['access'][me.DRAM]
            assert len(access_mem) == de.NUM
            access_runs = [a / float(s) for a, s
                           in zip(access_mem, layer_data_size)]
            total_nhops = [nh * r for nh, r in zip(unit_nhops, access_runs)]

            # Partition cost.
            cost_part = cost.noc_hop * sum(total_nhops)
            dict_part = {'unit_nhops': unit_nhops,
                         'total_nhops': total_nhops,
                         'part_lprev': part_lprev.as_pod_type(),
                         'part_lcurr': part_lcurr.as_pod_type()}

            # Combine.
            dict_loop.update({'cost': cost_loop})
            dict_part.update({'cost': cost_part})
            yield (cost_loop * resource.dim_nodes.size() + cost_part,
                   dict_loop, dict_part)

    return heapq.nsmallest(options.ntops, sweep(), key=lambda x: x[0])


def layer_schedule_search(layer, batch_size, resource, cost,
                          part_lprev, gen_nested_loop_desc, options):
    '''
    Search the best schedule for given layer and batch size.
    '''

    pool = multiprocessing.Pool(processes=options.nprocesses)
    results = []

    # Search NoC partition.
    for part_lcurr, layer_part in _get_partition2d_genfunc(options)(
            layer, resource.dim_nodes):

        unit_nhops = Partition.unit_nhops_layer_partition2d(
            layer, batch_size, part_lcurr, part_lprev)

        # Search loop blocking using partitioned layer.
        for nested_loop_desc in gen_nested_loop_desc(layer_part, batch_size,
                                                     resource.dim_array):

            r = pool.apply_async(_combine_search_lpbl_part2d,
                                 (resource, cost, nested_loop_desc, layer_part,
                                  batch_size, part_lcurr, part_lprev,
                                  unit_nhops, options))
            results.append(r)

    def retrieve_result():  # pylint: disable=missing-docstring
        for r in results:
            nsmallest = r.get(timeout=3600)
            for t in nsmallest:
                yield t

    tops = heapq.nsmallest(options.ntops, retrieve_result(), key=lambda x: x[0])
    pool.close()

    return list(tops)


def schedule_search(layers, batch_size, resource, cost, gen_nested_loop_desc,
                    options):
    '''
    Search the best schedule for given network and batch size.
    '''

    aggr_tops = [(0, OrderedDict()) for _ in range(options.ntops)]

    # Assume the first layer input is fully fmap partitioned (image tiled).
    partition2d_all_ofmp = [0] * pe.NUM
    partition2d_all_ofmp[pe.OUTP] = PhyDim2(1, 1)
    partition2d_all_ofmp[pe.OFMP] = resource.dim_nodes

    # Keep all previous layer partition schemes appeared in the top schedules.
    # Explore all of them for next layer.
    part_lprev_list = [Partition2dScheme(range(pe.NUM), partition2d_all_ofmp)]
    # The corresponding indexes of schedules in aggr_tops for the previous layer
    # partition scheme.
    aggr_top_indexes_list = [range(options.ntops)]

    for name, layer in layers.items():

        new_aggr_tops = []

        for part_lprev, aggr_top_indexes in zip(part_lprev_list,
                                                aggr_top_indexes_list):
            try:
                # For each previous layer partition scheme, search top schedules
                # for the current layer.
                tops = layer_schedule_search(layer, batch_size, resource, cost,
                                             part_lprev, gen_nested_loop_desc,
                                             options)
            except Exception as e:
                sys.stderr.write('Failed when scheduling layer {}'.format(name))
                raise e

            # Append all the current layer top schedules to all the previous top
            # schedules with the matching partition scheme.
            for t_idx in range(options.ntops):
                if t_idx >= len(tops):
                    break
                # 2: dict_part.
                assert tops[t_idx][2]['part_lprev'] == part_lprev.as_pod_type()
                for at_idx in aggr_top_indexes:
                    new_schedule = aggr_tops[at_idx][1].copy()
                    new_schedule.update({name: tops[t_idx]})
                    atop = (aggr_tops[at_idx][0] + tops[t_idx][0], new_schedule)
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
            part_lprev_pod = aggr_tops[at_idx][1][name][2]['part_lcurr']
            part_lprev = Partition2dScheme(*part_lprev_pod)
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

