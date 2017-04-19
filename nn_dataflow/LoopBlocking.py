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
Loop blocking optimization.

Include loop blocking and reordering.

For our problem, only deal with nifm, nofm, and batch loops.
'''

import itertools

from . import DataCategoryEnum as de
from . import LoopBlockingSolver
from . import Util
from .LoopBlockingScheme import LoopBlockingScheme


def gen_loopblocking_gbuf_regf(nested_loop_desc, resource, options):
    '''
    Generator for loop blocking schemes.
    '''
    del resource, options  # unused

    for ti, to, tb, orders in itertools.product( \
            Util.factorize(nested_loop_desc.loopcnt_ifm, 3),
            Util.factorize(nested_loop_desc.loopcnt_ofm, 3),
            Util.factorize(nested_loop_desc.loopcnt_bat, 3),
            itertools.product([None],
                              itertools.permutations((de.IFM, de.OFM)),
                              [None],
                              itertools.permutations((de.IFM, de.OFM)))):
        yield ti, to, tb, orders


def gen_loopblocking(nested_loop_desc, resource, options):
    '''
    Generator for loop blocking.
    '''
    if options.sw_solve_loopblocking:
        gen = LoopBlockingSolver.gen_loopblocking_gbuf_regf
    else:
        gen = gen_loopblocking_gbuf_regf

    for ti, to, tb, orders in gen(nested_loop_desc, resource, options):
        lbs = LoopBlockingScheme(nested_loop_desc, ti, to, tb, orders,
                                 resource, options)
        if lbs.is_valid():
            yield lbs


