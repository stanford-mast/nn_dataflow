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

import numbers

from . import loop_enum as le
from .. import util
from .loop_blocking_scheme import LoopBlockingScheme

class SchedulingConstraint(util.ContentHashClass):
    '''
    Layer scheduling constraint, which constrains top loop blocking factors.
    '''

    def __init__(self, topbat=0, topifm=0, topofm=0, update_dict=None):
        '''
        `topbat`, `topifm`, `topofm` specify the top-level loop blocking
        factors.

        `update_dict` specifies lazily updated rules to refine the constraint
        with previous scheduling results. It should be a mapping, from previous
        layer name to a function which takes two arguments: self, and the
        SchedulingResult instance of that layer.
        '''
        if any(n < 0 or not isinstance(n, numbers.Integral)
               for n in [topbat, topifm, topofm]):
            raise ValueError('SchedulingConstraint: '
                             'constrained factors must be positive integers.')

        if not update_dict:
            update_dict = {}
        if not isinstance(update_dict, dict):
            raise TypeError('SchedulingConstraint: '
                            'update_dict must be a dict instance.')
        update_dict = util.HashableDict.fromdict(update_dict)
        for val in update_dict.values():
            if not callable(val):
                raise TypeError('SchedulingConstraint: '
                                'values in update_dict must be callable.')

        self.topbat = topbat
        self.topifm = topifm
        self.topofm = topofm
        self.update_dict = update_dict

    def is_valid_top_bl(self, top_bl_t, top_bl_ord):
        '''
        Whether the given `top_bl_t` and `top_bl_lpe` are valid with the
        constraint.
        '''
        if self.update_dict:
            raise ValueError('SchedulingConstraint: update_dict is not empty, '
                             'rules have not been updated.')

        if self.topbat and self.topbat != top_bl_t[le.BAT]:
            return False
        if self.topifm and self.topifm != top_bl_t[le.IFM]:
            return False
        if self.topofm and self.topofm != top_bl_t[le.OFM]:
            return False

        del top_bl_ord

        return True

    def is_valid_part(self, part):
        '''
        Whether the given `part` is valid with the constraint.
        '''
        # pylint: disable=unused-argument
        if self.update_dict:
            raise ValueError('SchedulingConstraint: update_dict is not empty, '
                             'rules have not been updated.')

        return True

    def filter_gen_ts(self, gen_tifm, gen_tofm, gen_tbat):
        ''' Get the filtered generators for loop blocking factors. '''
        return self._filter_gen(gen_tifm, self.topifm), \
                self._filter_gen(gen_tofm, self.topofm), \
                self._filter_gen(gen_tbat, self.topbat)

    def update_by_prev(self, prev_results):
        '''
        Based on the previous layer scheduling results `prev_results` as a
        mapping from previous layer name to SchedulingResult instance, use the
        rules specified by `update_dict` to update the constraint.
        '''
        for layer_name in self.update_dict:
            self.update_dict[layer_name](self, prev_results[layer_name])
        self.update_dict = util.HashableDict()  # clear updated rules.

    @staticmethod
    def _filter_gen(gen, topt=0):
        ''' Get a new generator which filters the top factor. '''
        for tpl in gen:
            if topt in (0, tpl[0]):
                yield tpl

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(['{}={}'.format(k, repr(v))
                       for k, v in self.__dict__.items()]))


class SchedulingConstraintLayerPipeline(SchedulingConstraint):
    '''
    Layer scheduling constraint for inter-layer pipelining.

    Constraint includes:
    - topbat: top BAT loop blocking factor, which decides the number of groups
      for batch pipelining. It must match between all layers in a pipeline
      segment.
    - topifm/topofm: top IFM/OFM blocking factor, which decides the number of
      groups for fmap data forwarding between adjacent spatial scheduled layers
      in a pipeline segment. It must match between forwarding
      source/destination layers.
    - fbifm/fbofm: whether to fully buffer the fmap data of the layer on-chip.
      It indicates the baseline double-buffering between pipelined layers.

    For loop orders, the BAT loop must be at the outermost for batch
    pipelining. Then the loop associated with the forwarded data (IFM or OFM)
    must follow at the second outermost. If a data category (IFM or OFM) is
    fully buffered, then the corresponding loop is a trivial loop, which can be
    at any where.
    '''

    def __init__(self, topbat=0, topifm=0, topofm=0, fbifm=False, fbofm=False,
                 update_dict=None):

        if fbifm:
            # Fully-buffered IFM <=> topifm = 1.
            if topifm not in (0, 1):
                raise ValueError('SchedulingConstraintLayerPipeline: '
                                 'fully-buffered IFM implies topifm = 1.')
            topifm = 1

        if fbofm:
            # Fully-buffered OFM <=> topofm = 1.
            if topofm not in (0, 1):
                raise ValueError('SchedulingConstraintLayerPipeline: '
                                 'fully-buffered OFM implies topofm = 1.')
            topofm = 1

        if topifm > 1 and topofm > 1:
            raise ValueError('SchedulingConstraintLayerPipeline: '
                             'impossible to have both topifm and topofm > 1, '
                             'at least one of IFM and OFM must be a trivial '
                             'loop (= 1) or not constrained (= 0).')

        super(SchedulingConstraintLayerPipeline, self).__init__(
            topbat=topbat, topifm=topifm, topofm=topofm,
            update_dict=update_dict)

    def is_valid_top_bl(self, top_bl_t, top_bl_ord):

        if not super(SchedulingConstraintLayerPipeline, self).is_valid_top_bl(
                top_bl_t, top_bl_ord):
            return False

        # Loop orders.
        # Ordered loops from outer to inner.
        ord_lpe = LoopBlockingScheme.ordered_loops(top_bl_t, top_bl_ord,
                                                   lpe_only=True)
        if self.topbat > 1:
            if ord_lpe.pop(0) != le.BAT:
                return False
        # topifm and topofm cannot trigger together.
        if self.topifm > 1:
            if ord_lpe.pop(0) != le.IFM:
                return False
        if self.topofm > 1:
            if ord_lpe.pop(0) != le.OFM:
                return False

        return True

