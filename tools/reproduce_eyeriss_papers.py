'''
Reproduce optimal schedule for Eyeriss paper.
'''

import argparse
import multiprocessing
import sys

from eyeriss_search import do_scheduling


class _Namespace(object):  # pylint: disable=too-few-public-methods
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(args):
    ''' Main function. '''

    hier_cost = [200, 6, 2, 1]

    if args.paper == 'isscc16':
        mapping_args = _Namespace(
            net='eyeriss_alex_net',
            batch=4,
            word=16,
            nodes=[1, 1],
            array=[12, 14],
            regf=530,  # (225 + 12 + 24) * 16b
            gbuf=108 * 1024,  # 108 kB
            op_cost=1,
            hier_cost=hier_cost,
            hop_cost=0,
            unit_static_cost=0,
            disable_bypass='i o f'.split(),
            solve_loopblocking=False,
            hybrid_partition2d=False,
            processes=args.processes)

        results = do_scheduling(mapping_args)
        mappings = results['mappings']
        for name, mapping in mappings.items():
            ops = mapping[1]['ops']
            access = [sum(a) for a in mapping[1]['access']]
            assert len(access) == 4
            print '{},{},{}'.format(name, ops,
                                    ','.join([str(a) for a in access]))

    elif args.paper == 'isca16':
        mapping_args = _Namespace(
            net='eyeriss_alex_net',
            batch=16,
            word=16,
            nodes=[1, 1],
            array=[16, 16],
            regf=512,
            gbuf=128 * 1024,
            op_cost=1,
            hier_cost=hier_cost,
            hop_cost=0,
            unit_static_cost=0,
            disable_bypass='i o f'.split(),
            solve_loopblocking=False,
            hybrid_partition2d=False,
            processes=args.processes)

        results = do_scheduling(mapping_args)
        mappings = results['mappings']
        for name, mapping in mappings.items():
            ops = mapping[1]['ops']
            access = [sum(a) for a in mapping[1]['access']]
            assert len(access) == 4
            cost_breakdown = [ops * 1] + [a * c for a, c
                                          in zip(access, hier_cost)]
            print '{},{}'.format(name,
                                 ','.join([str(c) for c in cost_breakdown]))

    else:
        raise ValueError('Unrecognized paper name')

    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()  # pylint: disable=invalid-name

    ap.add_argument('paper', choices=['isscc16', 'isca16'],
                    help='the paper in which the results to reproduce')

    ap.add_argument('-p', '--processes', type=int,
                    default=multiprocessing.cpu_count()/2,
                    help='Number of parallel processes to use for search.')

    sys.exit(main(ap.parse_args()))

