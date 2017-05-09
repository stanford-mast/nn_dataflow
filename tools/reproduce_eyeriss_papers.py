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

import argparse
import multiprocessing
import sys

from eyeriss_search import do_scheduling

class _Namespace(object):  # pylint: disable=too-few-public-methods
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(args):
    ''' Reproduce optimal schedule for Eyeriss paper. '''

    hier_cost = [200, 6, 2, 1]

    if args.paper == 'isscc16':
        mapping_args = _Namespace(
            net='alex_net',
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
            mem_type='2D',
            disable_bypass='i o f'.split(),
            solve_loopblocking=False,
            hybrid_partition=False,
            batch_partition=False,
            top=1,
            processes=args.processes)

        results = do_scheduling(mapping_args)
        mappings = results['mappings']

        for layer in ['conv{}'.format(i) for i in range(1, 6)] \
                + ['fc{}'.format(i) for i in range(1, 4)]:
            ops = 0
            access = [0] * 4

            for layer_part in mappings:
                if not layer_part.startswith(layer):
                    continue
                ops += mappings[layer_part].dict_loop['ops']
                access = [a1 + sum(a2) for a1, a2
                          in zip(access,
                                 mappings[layer_part].dict_loop['access'])]

            print '{},{},{}'.format(layer, ops,
                                    ','.join([str(a) for a in access]))

    elif args.paper == 'isca16':
        mapping_args = _Namespace(
            net='alex_net',
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
            mem_type='2D',
            disable_bypass='i o f'.split(),
            solve_loopblocking=False,
            hybrid_partition=False,
            batch_partition=False,
            top=1,
            processes=args.processes)

        results = do_scheduling(mapping_args)
        mappings = results['mappings']

        for layer in ['conv{}'.format(i) for i in range(1, 6)] \
                + ['fc{}'.format(i) for i in range(1, 4)]:
            ops = 0
            access = [0] * 4

            for layer_part in mappings:
                if not layer_part.startswith(layer):
                    continue
                ops += mappings[layer_part].dict_loop['ops']
                access = [a1 + sum(a2) for a1, a2
                          in zip(access,
                                 mappings[layer_part].dict_loop['access'])]

            cost_breakdown = [ops * 1] + [a * c for a, c
                                          in zip(access, hier_cost)]

            print '{},{}'.format(layer,
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

