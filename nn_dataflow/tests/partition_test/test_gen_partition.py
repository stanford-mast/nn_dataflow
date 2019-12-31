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

from nn_dataflow.core import NodeRegion
from nn_dataflow.core import ParallelEnum as pe
from nn_dataflow.core import PhyDim2
from nn_dataflow import util

from . import TestPartitionFixture

class TestGenPartition(TestPartitionFixture):
    ''' Tests for gen_partition function. '''

    def test_full_util(self):
        ''' Full utilization. '''
        for dnkey in self.dim_nodes:
            dim_nodes = self.dim_nodes[dnkey]

            for wlkey in self.layers:
                for optkey in self.options:

                    for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey,
                                                    optkey=optkey):
                        self.assertTupleEqual(part.dim(), dim_nodes)

    def test_part_apprdiv(self):
        ''' OUTP, OFMP, and INPP approximately dividable. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey):
                    dim_ofmp = part.dim(pe.OFMP)
                    sz_outp = part.size(pe.OUTP)
                    sz_inpp = part.size(pe.INPP)

                    self.assertTrue(util.approx_dividable(layer.hofm,
                                                          dim_ofmp.h))
                    self.assertTrue(util.approx_dividable(layer.wofm,
                                                          dim_ofmp.w))
                    self.assertTrue(util.approx_dividable(layer.nofm, sz_outp))
                    self.assertTrue(util.approx_dividable(layer.nifm, sz_inpp))

    def test_part_batch_div(self):
        ''' BATP dividable. '''
        for part in self._gen_partition(dnkey='LG'):
            self.assertEqual(self.batch_size % part.size(pe.BATP), 0)

        for part in self._gen_partition(dnkey='PRIME'):
            self.assertEqual(part.size(pe.BATP), 1)

    def test_localregionlayer_no_inpp(self):
        ''' LocalRegionLayer does not use INPP. '''
        for wlkey in ['POOL', 'LR']:

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey):
                    self.assertEqual(part.size(pe.INPP), 1)

    def test_no_batp(self):
        ''' No BATP. '''
        part_set = set()
        for part in self._gen_partition(optkey='NOBATP'):
            self.assertEqual(part.size(pe.BATP), 1)
            part_set.add(part)
        part_set_truth = set(part for part in self._gen_partition()
                             if part.size(pe.BATP) == 1)
        self.assertSetEqual(part_set, part_set_truth)

    def test_no_inpp(self):
        ''' No INPP. '''
        part_set = set()
        for part in self._gen_partition(optkey='NOINPP'):
            self.assertEqual(part.size(pe.INPP), 1)
            part_set.add(part)
        part_set_truth = set(part for part in self._gen_partition()
                             if part.size(pe.INPP) == 1)
        self.assertSetEqual(part_set, part_set_truth)

    def test_no_hybrid_conv(self):
        ''' No hybrid partition for CONV layer. '''
        part_set = set()
        for part in self._gen_partition(optkey='NOHYB'):
            self.assertEqual(part.size(pe.OUTP), 1)
            part_set.add(part)
        part_set_truth = set(part for part in self._gen_partition()
                             if part.size(pe.OUTP, pe.INPP) == 1)
        self.assertSetEqual(part_set, part_set_truth)

    def test_no_hybrid_fc(self):
        ''' No hybrid partition for FC layer. '''
        part_set = set()
        for part in self._gen_partition(wlkey='FC', optkey='NOHYB'):
            self.assertEqual(part.size(pe.OFMP), 1)
            part_set.add(part)
        part_set_truth = set(part for part in self._gen_partition()
                             if part.size(pe.OFMP, pe.INPP) == 1)
        self.assertSetEqual(part_set, part_set_truth)

    def test_no_same(self):
        ''' No same scheme in generated PartitionScheme. '''
        for wlkey in self.layers:

            for dnkey in self.dim_nodes:

                part_list = list(self._gen_partition(wlkey=wlkey, dnkey=dnkey))

                self.assertEqual(len(part_list), len(set(part_list)))

    def test_no_eqv(self):
        ''' No equivalence in generated PartitionScheme. '''
        for wlkey in self.layers:

            for dnkey in self.dim_nodes:

                part_list = list(self._gen_partition(wlkey=wlkey, dnkey=dnkey))
                mapping_list = [self._part_index_to_coord(part)
                                for part in part_list]

                for idx, mapping in enumerate(mapping_list):
                    # Transpose coordinates.
                    mapping_t = {}
                    for pindex, coord in mapping.items():
                        mapping_t[pindex] = PhyDim2(h=coord.w, w=coord.h)

                    for idx2 in range(idx):
                        self.assertNotEqual(mapping_list[idx2], mapping,
                                            'test_no_eqv: {} {}: '
                                            'found equivalence.'
                                            '\n{}\n{}'
                                            .format(wlkey, dnkey,
                                                    part_list[idx],
                                                    part_list[idx2]))
                        self.assertNotEqual(mapping_list[idx2], mapping_t,
                                            'test_no_eqv: {} {}: '
                                            'found transpose equivalene.'
                                            '\n{}\n{}'
                                            .format(wlkey, dnkey,
                                                    part_list[idx],
                                                    part_list[idx2]))

    def test_exhaust(self):
        ''' Exhaust all except for equivalence. '''
        for wlkey in self.layers:

            for dnkey in self.dim_nodes:

                part_list = list(self._gen_partition(wlkey=wlkey, dnkey=dnkey))
                mapping_list = [self._part_index_to_coord(part)
                                for part in part_list]
                seen = [False] * len(part_list)

                for part in self._gen_partition_full(wlkey=wlkey, dnkey=dnkey):

                    try:
                        # Generated.
                        seen[part_list.index(part)] = True
                    except ValueError:

                        # Find an equivalence.
                        mapping = self._part_index_to_coord(part)
                        try:
                            seen[mapping_list.index(mapping)] = True
                        except ValueError:

                            # Find a transposed equivalence.
                            mapping_t = {}
                            for pindex, coord in mapping.items():
                                mapping_t[pindex] = PhyDim2(h=coord.w,
                                                            w=coord.h)
                            self.assertIn(mapping_t, mapping_list,
                                          'test_exhaust: {}: {} is missing'
                                          .format(wlkey, part))
                            seen[mapping_list.index(mapping_t)] = True

                # Confirm that baseline generator gives a super set of that
                # from skipped generator.
                self.assertTrue(all(seen))

    def test_skip_ratio(self):
        ''' Skip equivalence ratio. '''
        r = 1. * len(list(self._gen_partition())) \
                / len(list(self._gen_partition_full()))
        self.assertLessEqual(r, 0.15)

    def test_guaranteed(self):
        ''' Guaranteed. '''
        optkey = 'NOBATP'

        for wlkey in self.layers:
            if wlkey.startswith('SSM'):
                print(wlkey)
                self.assertEqual(len(list(self._gen_partition(wlkey=wlkey,
                                                              optkey=optkey))),
                                 0)

                part_list = list(self._gen_partition(wlkey=wlkey,
                                                     optkey=optkey,
                                                     guaranteed=True))
                self.assertEqual(len(part_list), 1)
                part = part_list[0]
                self.assertEqual(part.size(pe.BATP), 1)
                self.assertEqual(part.size(pe.INPP), 1)
                self.assertTrue(part.size(pe.OUTP) == 1
                                or part.size(pe.OFMP) == 1)

    def _part_index_to_coord(self, part):
        ''' Get the mapping from partition index to coordinate. '''
        nr = NodeRegion(origin=PhyDim2(0, 0), dim=part.dim(),
                        type=NodeRegion.PROC)
        mapping = {}
        for pidx in part.gen_pidx():
            coord = part.coordinate(nr, pidx)

            assert len(pidx) == 4
            pindex = (pidx[pe.OFMP].h, pidx[pe.OFMP].w,
                      pidx[pe.OUTP].h * part.dim(pe.OUTP).w + pidx[pe.OUTP].w,
                      pidx[pe.INPP].h * part.dim(pe.INPP).w + pidx[pe.INPP].w,
                      pidx[pe.BATP].h * part.dim(pe.BATP).w + pidx[pe.BATP].w)
            self.assertNotIn(pindex, mapping)
            mapping[pindex] = coord

        return mapping

