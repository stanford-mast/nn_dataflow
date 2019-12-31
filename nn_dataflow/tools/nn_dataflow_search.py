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

import argparse
import json
import multiprocessing
import sys
import time
from collections import OrderedDict

from nn_dataflow.core import NNDataflow
from nn_dataflow.core import Cost
from nn_dataflow.core import DataCategoryEnum as de
from nn_dataflow.core import MapStrategyEyeriss
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import Option
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource

from nn_dataflow.nns import all_networks
from nn_dataflow.nns import import_network

from nn_dataflow.version import get_version

def stats_dict(dfsch, cost):
    '''
    Get the stats as an OrderedDict from the NNDataflowScheme.
    '''
    stats = OrderedDict()

    ## Basic stats.

    stats['total_cost'] = dfsch.total_cost
    stats['total_time'] = dfsch.total_time

    stats['total_ops'] = dfsch.total_ops
    stats['total_accesses'] = dfsch.total_accesses
    stats['total_noc_hops'] = dfsch.total_noc_hops

    ## Cost breakdown.

    total_op_cost = dfsch.total_ops * cost.mac_op
    total_access_cost = sum(a * c for a, c
                            in zip(dfsch.total_accesses, cost.mem_hier))
    total_noc_cost = dfsch.total_noc_hops * cost.noc_hop
    total_static_cost = dfsch.total_time * cost.idl_unit

    sum_cost = total_op_cost + total_access_cost + total_noc_cost \
            + total_static_cost
    assert abs(sum_cost / dfsch.total_cost - 1) < 0.001

    stats['total_op_cost'] = total_op_cost
    stats['total_access_cost'] = total_access_cost
    stats['total_noc_cost'] = total_noc_cost
    stats['total_static_cost'] = total_static_cost

    ## Other stats.

    stats['active_node_pes'] = dfsch.perlayer_stats('active_node_pes')
    stats['dram_bandwidth'] = dfsch.perlayer_stats('dram_bandwidth')
    stats['segment_time'] = dfsch.segment_time_list()
    stats['segment_dram_time'] = dfsch.segment_dram_time_list()
    stats['input_layout'] = dfsch.input_layout
    stats['ext_layout_dict'] = dfsch.ext_layout_dict
    stats['schedules'] = dfsch.res_dict

    return stats


def do_scheduling(args):
    '''
    Get optimal scheduling for given problem. Return a result schedule.
    '''

    ## Network.

    network = import_network(args.net)
    batch_size = args.batch

    ## Resource.

    dim_nodes = PhyDim2(*args.nodes)
    dim_array = PhyDim2(*args.array)

    # Sizes of gbuf and regf are in words.
    word = (args.word + 7) // 8
    size_gbuf = args.gbuf // word
    size_regf = args.regf // word

    array_bus_width = args.bus_width // args.word
    if not array_bus_width:
        array_bus_width = float('inf')
    dram_bandwidth = args.dram_bw / word

    proc_region = NodeRegion(dim=dim_nodes,
                             origin=PhyDim2(0, 0),
                             type=NodeRegion.PROC)

    if args.mem_type == '2D':
        # Memory nodes are on two sides.
        data_region = NodeRegion(dim=PhyDim2(2, 2),
                                 origin=PhyDim2(0, 0),
                                 dist=dim_nodes - PhyDim2(1, 1),
                                 type=NodeRegion.DRAM)
        assert data_region.rel2abs(PhyDim2(1, 1)) + PhyDim2(1, 1) \
                == proc_region.dim
    elif args.mem_type == '3D':
        # Memory nodes are on the top.
        data_region = NodeRegion(dim=dim_nodes,
                                 origin=PhyDim2(0, 0),
                                 type=NodeRegion.DRAM)

    resource = Resource(proc_region=proc_region,
                        dram_region=data_region,
                        src_data_region=data_region,
                        dst_data_region=data_region,
                        dim_array=dim_array,
                        size_gbuf=size_gbuf,
                        size_regf=size_regf,
                        array_bus_width=array_bus_width,
                        dram_bandwidth=dram_bandwidth,
                        no_time_mux=False)

    ## Cost.

    hier_cost = [0] * me.NUM
    hier_cost[me.DRAM] = args.hier_cost[0]
    hier_cost[me.GBUF] = args.hier_cost[1]
    hier_cost[me.ITCN] = args.hier_cost[2]
    hier_cost[me.REGF] = args.hier_cost[3]
    cost = Cost(mac_op=args.op_cost,
                mem_hier=tuple(hier_cost),
                noc_hop=args.hop_cost,
                idl_unit=args.unit_idle_cost)

    ## Options.

    bypass = [True] * de.NUM
    bypass[de.IFM] = 'i' not in args.disable_bypass
    bypass[de.OFM] = 'o' not in args.disable_bypass
    bypass[de.FIL] = 'f' not in args.disable_bypass
    options = Option(sw_gbuf_bypass=tuple(bypass),
                     sw_solve_loopblocking=args.solve_loopblocking,
                     hw_access_forwarding=args.enable_access_forwarding,
                     hw_gbuf_sharing=args.enable_gbuf_sharing,
                     hw_gbuf_save_writeback=args.enable_save_writeback,
                     partition_hybrid=args.hybrid_partition,
                     partition_batch=args.batch_partition,
                     partition_ifmaps=args.ifmaps_partition,
                     partition_interlayer=args.interlayer_partition,
                     layer_pipeline_time_ovhd=args.layer_pipeline_time_overhead,
                     layer_pipeline_max_degree=args.layer_pipeline_max_degree,
                     layer_pipeline_opt=not args.disable_interlayer_opt,
                     opt_goal=args.goal.lower(),
                     ntops=args.top,
                     nprocesses=args.processes,
                     verbose=args.verbose)

    ## Search schedules.

    nnd = NNDataflow(network, batch_size, resource, cost, MapStrategyEyeriss)
    tbeg = time.time()
    tops, cache_stats = nnd.schedule_search(options)
    tend = time.time()
    telapsed = tend - tbeg

    if not tops:
        sys.stderr.write('No valid dataflow found.\n')
        return None

    top = tops[0]

    ## Write results.

    res_map = OrderedDict()

    res_map['version'] = get_version(with_local=True)

    res_map['net'] = args.net
    res_map['batch'] = args.batch

    res_map['resource'] = resource._asdict()
    res_map['cost'] = cost._asdict()
    res_map['options'] = options._asdict()

    res_map['cache_stats'] = cache_stats
    res_map['elapsed'] = telapsed

    stats = stats_dict(top, cost)
    for key, val in stats.items():
        res_map[key] = val

    return res_map


def argparser():
    ''' Argument parser. '''

    ap = argparse.ArgumentParser()

    ap.add_argument('net',
                    help='network name, should be a .py file under "nns". '
                         'Choices: {}.'.format(', '.join(all_networks())))

    ap.add_argument('--batch', type=int, required=True,
                    help='batch size')
    ap.add_argument('--word', type=int, default=16,
                    help='word size in bits')

    ap.add_argument('--nodes', type=int, nargs=2, required=True,
                    metavar=('H', 'W'),
                    help='Parallel node partitioning dimensions')
    ap.add_argument('--array', type=int, nargs=2, required=True,
                    metavar=('H', 'W'),
                    help='PE array dimensions')

    ap.add_argument('--regf', type=int, required=True,
                    help='register file size in bytes per PE')
    ap.add_argument('--gbuf', type=int, required=True,
                    help='global buffer size in bytes')

    ap.add_argument('--bus-width', type=int, default=0,
                    help='array bus width in bits. set 0 to ignore')
    ap.add_argument('--dram-bw', type=float, default='inf',
                    help='total DRAM bandwidth in bytes per cycle.')

    ap.add_argument('--op-cost', type=float, default=1,
                    help='cost of arithmetic operation')
    ap.add_argument('--hier-cost', type=float, nargs=4, default=[200, 6, 2, 1],
                    metavar=('DRAM_COST', 'GBUF_COST', 'ITCN_COST',
                             'REGF_COST'),
                    help='cost of access to memory hierarchy')
    ap.add_argument('--hop-cost', type=float, default=10,
                    help='cost of access through one NoC hop')
    ap.add_argument('--unit-idle-cost', type=float, default=0,
                    help='static cost over all nodes for unit execution time')

    ap.add_argument('--mem-type', default='2D', choices=['2D', '3D'],
                    help='memory type. "2D" has memory only on edge nodes; '
                         '"3D" has memory vertially on top of all nodes.')

    ap.add_argument('--disable-bypass', nargs='*', default=[],
                    choices=['i', 'o', 'f'],
                    help='whether disallowing gbuf bypass for i (input), o '
                         '(output), or f (filter)')
    ap.add_argument('--solve-loopblocking', action='store_true',
                    help='Use analytical solver to choose loop blocking. '
                         'Otherwise use exhaustive search.')
    ap.add_argument('--enable-access-forwarding', action='store_true',
                    help='Each node fetches a subset of data and forwards to '
                         'other nodes.')
    ap.add_argument('--enable-gbuf-sharing', action='store_true',
                    help='Share gbuf capacity across nodes through NoC.')
    ap.add_argument('--enable-save-writeback', action='store_true',
                    help='Allow to save the writeback to memory for the '
                         'intermediate data between layers if able to '
                         'store the entire data set in on-chip buffers.')
    ap.add_argument('--disable-interlayer-opt',
                    '--basic-interlayer-partition',
                    action='store_true',
                    help='Disable optimizations and only allow basic '
                         'inter-layer pipeline.')

    ap.add_argument('--hybrid-partition',
                    '--hybrid-partition2d',  # deprecated old name
                    action='store_true',
                    help='Use hybrid partition for layer for node mapping. '
                         'Otherwise use naive method based on layer type.')
    ap.add_argument('--batch-partition', action='store_true',
                    help='Allow partitioning batch, i.e., consider data '
                         'parallelism.')
    ap.add_argument('--ifmaps-partition', '--ifmap-partition',
                    action='store_true',
                    help='Allow partitioning ifmap channel dimension, which '
                         'requires extra data synchronization.')
    ap.add_argument('--interlayer-partition', '--inter-layer-partition',
                    action='store_true',
                    help='Allow partitioning resources across multiple layers '
                         'and process them simultaneously as an inter-layer '
                         'pipeline.')

    ap.add_argument('--layer-pipeline-time-overhead',
                    type=float, default=float('inf'),
                    help='maximum allowed execution time overhead due to '
                         'layer pipelining.')
    ap.add_argument('--layer-pipeline-max-degree',
                    type=float, default=float('inf'),
                    help='maximum allowed layer pipelining degree, i.e., '
                         'number of vertices in a pipeline segment.')

    ap.add_argument('-g', '--goal', default='e',
                    choices=['e', 'd', 'ed', 'E', 'D', 'ED'],
                    help='Goal of optimization: E(nergy), D(elay), or ED.')
    ap.add_argument('-t', '--top', type=int, default=1,
                    help='Number of top schedules to keep during search.')
    ap.add_argument('-p', '--processes', type=int,
                    default=multiprocessing.cpu_count()//2,
                    help='Number of parallel processes to use for search.')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='Show progress and details.')

    return ap


def main():
    ''' Main function. '''
    args = argparser().parse_args()
    res = do_scheduling(args)
    json.dump(res, sys.stdout, indent=2, default=lambda _: None)
    sys.stdout.write('\n')
    return 0 if res else 2


if __name__ == '__main__':
    sys.exit(main())

