'''
Hardware resource specification.
'''

from collections import namedtuple

Resource = namedtuple('Resource',
                      ['dim_nodes', 'dim_array', 'size_gbuf', 'size_regf'])

