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
import sys

from nn_dataflow.core import ConvLayer, FCLayer

from nn_dataflow.nns import import_network

KILO = 1024.
MILLION = 1024.*1024.

STR_FMT_NAME_LEN = '30'
STR_FMT_NUMB_LEN = '12'
STR_FMT_NUMB_PCS = '2'

STR_FMT_NAME = '{:' + STR_FMT_NAME_LEN + 's}'
STR_FMT_NUMB_HDER = '{:>' + STR_FMT_NUMB_LEN + '}'
STR_FMT_NUMB = '{:' + STR_FMT_NUMB_LEN + '.' + STR_FMT_NUMB_PCS + 'f}'

def layer_stats(args):
    ''' Print stats of layers in the network. '''

    network = import_network(args.net)
    word_bytes = (args.word + 7) // 8
    batch = args.batch

    hder_fmt = ','.join([STR_FMT_NAME] + [STR_FMT_NUMB_HDER] * 5) + '\n'
    line_fmt = ','.join([STR_FMT_NAME] + [STR_FMT_NUMB] * 5) + '\n'
    line_sep = '-' * int(STR_FMT_NAME_LEN) + '\n'

    # Header.
    sys.stdout.write(hder_fmt
                     .format('Layer',
                             'Ifmap/kB', 'Ofmap/kB', 'Weight/kB',
                             'MACs/M', 'MinOptBuf/kB'))

    # Aggregate stats.
    max_fmaps = 0
    max_filters = 0
    max_ops = 0
    sum_fmaps = 0
    sum_filters = 0
    sum_ops = 0
    convs = 0
    fcs = 0

    for name in network:

        layer = network[name]

        if isinstance(layer, FCLayer):
            fcs += 1
        elif isinstance(layer, ConvLayer):
            convs += 1

        ifmap_size = layer.total_ifmap_size(word_bytes) * batch / KILO
        ofmap_size = layer.total_ofmap_size(word_bytes) * batch / KILO
        try:
            filter_size = layer.total_filter_size(word_bytes) / KILO
        except AttributeError:
            filter_size = 0

        ops = layer.total_ops(batch) / MILLION

        # The minimum optimal buffer size is the sum of the full size (two
        # dimensions) for one data category, the size of one dimension for the
        # second, and the size of one point for the third.
        min_opt_buf_size = min(
            filter_size + (ifmap_size + ofmap_size / layer.nofm) / batch,
            filter_size + (ifmap_size / layer.nifm + ofmap_size) / batch,
            ifmap_size + (ofmap_size + filter_size / layer.nifm) / layer.nofm,
            ifmap_size + (ofmap_size / batch + filter_size) / layer.nofm,
            ofmap_size + (ifmap_size + filter_size / layer.nofm) / layer.nifm,
            ofmap_size + (ifmap_size / batch + filter_size) / layer.nifm)

        sys.stdout.write(line_fmt
                         .format(name,
                                 ifmap_size, ofmap_size, filter_size,
                                 ops, min_opt_buf_size))

        max_fmaps = max(max_fmaps, ofmap_size)
        max_filters = max(max_filters, filter_size)
        max_ops = max(max_ops, ops)
        sum_fmaps += ofmap_size
        sum_filters += filter_size
        sum_ops += ops

    sys.stdout.write(line_sep)

    sys.stdout.write(line_fmt
                     .format('MAX',
                             float('nan'), max_fmaps, max_filters,
                             max_ops, float('nan')))
    sys.stdout.write(line_fmt
                     .format('SUM',
                             float('nan'), sum_fmaps, sum_filters,
                             sum_ops, float('nan')))

    sys.stdout.write(line_sep)

    sys.stdout.write('# CONV layers = {}, # FC layers = {}\n'
                     .format(convs, fcs))


def argparser():
    ''' Argument parser. '''

    ap = argparse.ArgumentParser()

    ap.add_argument('net',
                    help='network name, should be a .py file under examples')

    ap.add_argument('-b', '--batch', type=int, default=1,
                    help='batch size')
    ap.add_argument('-w', '--word', type=int, default=16,
                    help='word size in bits')

    return ap


if __name__ == '__main__':
    layer_stats(argparser().parse_args())

