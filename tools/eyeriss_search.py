'''
Search optimal schedule and partitioning.
'''

import argparse
import json
import multiprocessing
import numpy as np
import sys
from collections import OrderedDict

from nn_dataflow import Cost
from nn_dataflow import DataCategoryEnum as de
from nn_dataflow import MemHierEnum as me
from nn_dataflow import Option
from nn_dataflow import PhyDim2
from nn_dataflow import Resource
from nn_dataflow import schedule_search

from nn_dataflow import MapEyeriss

from examples import import_network_layers


def do_scheduling(args):
    '''
    Get optimal scheduling for given problem. Return a result schedule.
    '''

    layers = import_network_layers(args.net)

    batch_size = args.batch
    word = (args.word + 7) / 8

    resource = Resource(dim_nodes=PhyDim2(*args.nodes),
                        dim_array=PhyDim2(*args.array),
                        size_gbuf=args.gbuf/word,
                        size_regf=args.regf/word*np.prod(args.array))

    hier_cost = [0] * me.NUM
    hier_cost[me.DRAM] = args.hier_cost[0]
    hier_cost[me.GBUF] = args.hier_cost[1]
    hier_cost[me.ITCN] = args.hier_cost[2]
    hier_cost[me.REGF] = args.hier_cost[3]
    cost = Cost(cost_memhier=hier_cost,
                cost_nochop=args.hop_cost,
                cost_macop=args.op_cost,
                cost_unit_static=args.unit_static_cost)

    bypass = [True] * de.NUM
    bypass[de.IFM] = 'i' not in args.disable_bypass
    bypass[de.OFM] = 'o' not in args.disable_bypass
    bypass[de.FIL] = 'f' not in args.disable_bypass
    options = Option(allow_gbuf_bypass=bypass,
                     solve_loopblocking=args.solve_loopblocking,
                     hybrid_partition2d=args.hybrid_partition2d,
                     ntops=1,
                     nprocesses=args.processes)

    # Search schedules.
    tops = schedule_search(layers, batch_size, resource, cost,
                           MapEyeriss.gen_nested_loop_desc, options)

    top_mapping = tops[0]

    # Get stats.
    stats = {}
    stats['total_cost'] = top_mapping[0]

    stats['total_time'] = 0
    stats['total_noc_cost'] = 0
    stats['total_ops_per_node'] = 0
    stats['max_dram_bw_per_node'] = 0
    stats['max_dram_bw_layer'] = None
    stats['total_accesses_per_node'] = [0] * me.NUM
    for name in layers.keys():
        layer_top_mapping = top_mapping[1][name]
        layer_dict_loop = layer_top_mapping[1]
        layer_dict_part = layer_top_mapping[2]

        stats['total_time'] += layer_dict_loop['time']
        stats['total_noc_cost'] += layer_dict_part['cost']
        stats['total_ops_per_node'] += layer_dict_loop['ops']
        dram_bw_per_node = sum(layer_dict_loop['access'][me.DRAM]) \
                / float(layer_dict_loop['time'])
        if dram_bw_per_node > stats['max_dram_bw_per_node']:
            stats['max_dram_bw_per_node'] = dram_bw_per_node
            stats['max_dram_bw_layer'] = name
        stats['total_accesses_per_node'] = [
            s + a for s, a in zip(stats['total_accesses_per_node'],
                                  [sum(alist) for alist
                                   in layer_dict_loop['access']])]

    stats['average_active_pes'] = stats['total_ops_per_node'] \
            / float(stats['total_time'])
    stats['total_static_cost'] = stats['total_time'] * cost.unit_static() \
            * resource.dim_nodes.size()

    sum_cost = 0
    num_nodes = resource.dim_nodes.size()
    sum_cost += stats['total_ops_per_node'] * num_nodes * cost.macop()
    sum_cost += sum([a * c * num_nodes
                     for a, c in zip(stats['total_accesses_per_node'],
                                     cost.memhier())])
    sum_cost += stats['total_static_cost']
    sum_cost += stats['total_noc_cost']
    assert abs(sum_cost / stats['total_cost'] - 1) < 0.001

    # Write results.
    res_map = OrderedDict()
    for argname in ['net', 'batch', 'word', 'nodes', 'array', 'regf', 'gbuf',
                    'op_cost', 'hier_cost', 'hop_cost', 'unit_static_cost',
                    'solve_loopblocking', 'hybrid_partition2d',
                    'disable_bypass']:
        res_map[argname] = getattr(args, argname)
    for statname in ['total_time', 'total_cost', 'total_static_cost',
                     'total_noc_cost', 'average_active_pes',
                     'max_dram_bw_per_node', 'max_dram_bw_layer',
                     'total_ops_per_node', 'total_accesses_per_node']:
        res_map[statname] = stats[statname]
    res_map['mappings'] = top_mapping[1]

    return res_map


def main(args):
    ''' Main function. '''
    json.dump(do_scheduling(args), sys.stdout, indent=2)
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()  # pylint: disable=invalid-name

    ap.add_argument('net',
                    help='network name, should be a .py file under examples')

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

    ap.add_argument('--op-cost', type=float, default=1,
                    help='cost of arithmetic operation')
    ap.add_argument('--hier-cost', type=float, nargs=4, default=[200, 6, 2, 1],
                    metavar=('DRAM_COST', 'GBUF_COST', 'ITCN_COST',
                             'REGF_COST'),
                    help='cost of access to memory hierarchy')
    ap.add_argument('--hop-cost', type=float, default=100,
                    help='cost of access through one NoC hop')
    ap.add_argument('--unit-static-cost', type=float, default=0,
                    help='static cost for unit execution time')

    ap.add_argument('--disable-bypass', nargs='*', default=[],
                    choices=['i', 'o', 'f'],
                    help='whether disallowing gbuf bypass for i (input), o '
                         '(output), or f (filter)')

    ap.add_argument('--solve-loopblocking', action='store_true',
                    help='Use analytical solver to choose loop blocking. '
                         'Otherwise use exhaustive search.')
    ap.add_argument('--hybrid-partition2d', action='store_true',
                    help='Use hybrid partition for layer for node mapping. '
                         'Otherwise use naive method based on layer type.')

    ap.add_argument('-p', '--processes', type=int,
                    default=multiprocessing.cpu_count()/2,
                    help='Number of parallel processes to use for search.')

    sys.exit(main(ap.parse_args()))

