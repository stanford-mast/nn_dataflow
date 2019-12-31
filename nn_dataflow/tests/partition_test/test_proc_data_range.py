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

from collections import Counter

from nn_dataflow.core import partition
from nn_dataflow.core import ConvLayer, LocalRegionLayer
from nn_dataflow.core import FmapRange, FmapRangeMap
from nn_dataflow.core import ParallelEnum as pe

from . import TestPartitionFixture

class TestProcDataRange(TestPartitionFixture):
    ''' Tests for proc_data_range functions. '''

    def test_io_count(self):
        ''' i/ofmap count. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey):

                    i_frngs = []
                    o_frngs = []

                    for pidx in part.gen_pidx():
                        _, ifrng, ofrng = partition.proc_data_range(
                            layer, self.batch_size, part, pidx)

                        i_frngs.append(ifrng)
                        o_frngs.append(ofrng)

                    i_cnts = Counter(i_frngs)
                    o_cnts = Counter(o_frngs)

                    if isinstance(layer, ConvLayer):
                        pidx_per_rng = part.size(pe.OUTP)
                    elif isinstance(layer, LocalRegionLayer):
                        pidx_per_rng = 1

                    self.assertEqual(len(i_cnts), part.size() // pidx_per_rng)
                    for v in i_cnts.values():
                        self.assertEqual(v, pidx_per_rng)

                    self.assertEqual(len(o_cnts),
                                     part.size() // part.size(pe.INPP))
                    for v in o_cnts.values():
                        self.assertEqual(v, part.size(pe.INPP))

    def test_o_no_overlap(self):
        ''' ofmap no overlap. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey,
                                                optkey='NOINPP'):

                    fr_list = []

                    for pidx in part.gen_pidx():

                        _, _, fr = partition.proc_data_range(
                            layer, self.batch_size, part, pidx)

                        for fr2 in fr_list:
                            self.assertEqual(fr.overlap(fr2).size(), 0,
                                             'test_o_no_overlap: {}: '
                                             '{} and {} overlap.'
                                             .format(wlkey, fr, fr2))

                        fr_list.append(fr)

    def test_io_full_layer(self):
        ''' i/ofmap full layer. '''
        for wlkey in ['SM', 'POOL']:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey,
                                                optkey='NOINPP'):

                    # Remove ifmap point from full set.
                    ifp_set = set(FmapRange(fp_beg=(0, 0, 0, 0),
                                            fp_end=(self.batch_size,
                                                    layer.nifm,
                                                    layer.hifm,
                                                    layer.wifm)).range())
                    # Add ofmap ranges to a map.
                    ofrmap = FmapRangeMap()

                    for pidx in part.gen_pidx():

                        _, ifrng, ofrng = partition.proc_data_range(
                            layer, self.batch_size, part, pidx)

                        for ifp in ifrng.range():
                            ifp_set.discard(ifp)

                        ofrmap.add(ofrng, 0)

                    # Ifmap point set should be empty now.
                    self.assertFalse(ifp_set)

                    # Ofmap range map should be full now.
                    self.assertTrue(ofrmap.is_complete())
                    cfr = ofrmap.complete_fmap_range()
                    self.assertEqual(cfr.size('b'), self.batch_size)
                    self.assertEqual(cfr.size('n'), layer.nofm)
                    self.assertEqual(cfr.size('h'), layer.hofm)
                    self.assertEqual(cfr.size('w'), layer.wofm)

    def test_io_equal_size(self):
        ''' i/ofmap approximately equal range size. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey,
                                                optkey='NOINPP'):

                    ifr_list = []
                    osz_list = []

                    for pidx in part.gen_pidx():

                        _, ifrng, ofrng = partition.proc_data_range(
                            layer, self.batch_size, part, pidx)

                        ifr_list.append(ifrng)
                        osz_list.append([ofrng.size(d) for d in 'bnhw'])

                    # For ifmaps.
                    for d in 'bnhw':
                        dsz_list = [ifrng.size(d) for ifrng in ifr_list]

                        thr = 1
                        if isinstance(layer, LocalRegionLayer):
                            thr = layer.nreg - layer.nreg // 2 \
                                    if d == 'n' else \
                                    (layer.hreg if d == 'h'
                                     else (layer.wreg if d == 'w'
                                           else 1))

                        self.assertEqual(len(dsz_list), part.size())
                        self.assertLessEqual(max(dsz_list) - min(dsz_list),
                                             thr,
                                             'test_i_equal_size: {} {}: '
                                             'dim {} range size diverges. '
                                             'max {} min {}'
                                             .format(wlkey, dnkey, d,
                                                     max(dsz_list),
                                                     min(dsz_list)))

                    # For ofmaps.
                    for dsz_list, d in zip(zip(*osz_list), 'bnhw'):
                        self.assertEqual(len(dsz_list), part.size())
                        self.assertLessEqual(max(dsz_list) - min(dsz_list), 1,
                                             'test_o_equal_size: {} {}: '
                                             'dim {} range size diverges. '
                                             'max {} min {}'
                                             .format(wlkey, dnkey, d,
                                                     max(dsz_list),
                                                     min(dsz_list)))

    def test_match_io_fmap_range(self):
        ''' ofmap and ifmap range match. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey,
                                                optkey='NOINPP'):

                    for pidx in part.gen_pidx():

                        _, ifrng, ofrng = partition.proc_data_range(
                            layer, self.batch_size, part, pidx)

                        self.assertEqual(ofrng.size('b'), ifrng.size('b'))

                        if isinstance(layer, ConvLayer):
                            ol = ConvLayer(nifm=ifrng.size('n'),
                                           nofm=ofrng.size('n'),
                                           sofm=(ofrng.size('h'),
                                                 ofrng.size('w')),
                                           sfil=(layer.hfil, layer.wfil),
                                           strd=(layer.htrd, layer.wtrd))
                            il = ol.input_layer()
                            self.assertEqual(il.nofm, ifrng.size('n'))
                        elif isinstance(layer, LocalRegionLayer):
                            nofm_beg, nofm_end = ofrng.beg_end('n')
                            nifm_beg, nifm_end = ifrng.beg_end('n')
                            self.assertEqual(nifm_beg, max(0, \
                                    nofm_beg - layer.nreg // 2))
                            self.assertEqual(nifm_end, min(layer.nifm, \
                                    nofm_end + layer.nreg - layer.nreg // 2))

                            ol = LocalRegionLayer(nofm=ofrng.size('n'),
                                                  sofm=(ofrng.size('h'),
                                                        ofrng.size('w')),
                                                  nreg=layer.nreg,
                                                  sreg=(layer.hreg,
                                                        layer.wreg),
                                                  strd=(layer.htrd,
                                                        layer.wtrd))
                            il = ol.input_layer()

                        self.assertEqual(il.hofm, ifrng.size('h'))
                        self.assertEqual(il.wofm, ifrng.size('w'))

    def test_filrng(self):
        ''' Filter ranges. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey):

                    for pidx in part.gen_pidx():
                        filrng, ifrng, ofrng = partition.proc_data_range(
                            layer, self.batch_size, part, pidx)

                        self.assertEqual(len(filrng), 2)
                        if isinstance(layer, ConvLayer):
                            self.assertEqual(filrng[0].size(), ifrng.size('n'))
                            self.assertEqual(filrng[1].size(), ofrng.size('n'))
                        elif isinstance(layer, LocalRegionLayer):
                            self.assertTrue(filrng[0].empty())
                            self.assertTrue(filrng[1].empty())

