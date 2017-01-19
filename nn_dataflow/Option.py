'''
Schedule options for nn_dataflow.
'''

from collections import namedtuple


OPTION_LIST = ['allow_gbuf_bypass',
               'solve_loopblocking',
               'hybrid_partition2d',
               'ntops',
               'nprocesses',
              ]
Option = namedtuple('Option', OPTION_LIST)

