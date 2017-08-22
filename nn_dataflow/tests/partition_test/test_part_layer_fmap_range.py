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

from collections import Counter

from nn_dataflow.core import partition
from nn_dataflow.core import ConvLayer, LocalRegionLayer
from nn_dataflow.core import FmapRange, FmapRangeMap
from nn_dataflow.core import ParallelEnum as pe

from . import TestPartitionFixture

class TestPartLayerFmapRange(TestPartitionFixture):
    ''' Tests for part_layer_i/ofmap_range functions. '''

    def test_o_count(self):
        ''' ofmap count. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey):

                    cnts = Counter(partition.part_layer_ofmap_range(
                        layer, self.batch_size, part, pidx)
                                   for pidx in part.gen_pidx())

                    self.assertEqual(len(cnts),
                                     part.size() // part.size(pe.INPP))
                    for v in cnts.values():
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

                        fr = partition.part_layer_ofmap_range(
                            layer, self.batch_size, part, pidx)

                        for fr2 in fr_list:
                            self.assertEqual(fr.overlap(fr2).size(), 0,
                                             'test_o_no_overlap: {}: '
                                             '{} and {} overlap.'
                                             .format(wlkey, fr, fr2))

                        fr_list.append(fr)

    def test_o_full_layer(self):
        ''' ofmap full layer. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey,
                                                optkey='NOINPP'):

                    frmap = FmapRangeMap()

                    for pidx in part.gen_pidx():

                        fr = partition.part_layer_ofmap_range(
                            layer, self.batch_size, part, pidx)
                        frmap.add(fr, 0)

                    self.assertTrue(frmap.is_complete())

                    cfr = frmap.complete_fmap_range()
                    self.assertEqual(cfr.size('b'), self.batch_size)
                    self.assertEqual(cfr.size('n'), layer.nofm)
                    self.assertEqual(cfr.size('h'), layer.hofm)
                    self.assertEqual(cfr.size('w'), layer.wofm)

    def test_o_equal_size(self):
        ''' ofmap approximately equal range size. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey,
                                                optkey='NOINPP'):

                    size_list = []

                    for pidx in part.gen_pidx():

                        fr = partition.part_layer_ofmap_range(
                            layer, self.batch_size, part, pidx)

                        size_list.append([fr.size(d) for d in 'bnhw'])

                    for dsz_list, d in zip(zip(*size_list), 'bnhw'):
                        self.assertEqual(len(dsz_list), part.size())
                        self.assertLessEqual(max(dsz_list) - min(dsz_list), 1,
                                             'test_o_equal_size: {} {}: '
                                             'dim {} range size diverges. '
                                             'max {} min {}'
                                             .format(wlkey, dnkey, d,
                                                     max(dsz_list),
                                                     min(dsz_list)))

    def test_i_count(self):
        ''' ifmap count. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey):

                    cnts = Counter(partition.part_layer_ifmap_range(
                        layer, self.batch_size, part, pidx)
                                   for pidx in part.gen_pidx())

                    if isinstance(layer, ConvLayer):
                        pidx_per_rng = part.size(pe.OUTP)
                    elif isinstance(layer, LocalRegionLayer):
                        pidx_per_rng = 1

                    self.assertEqual(len(cnts), part.size() // pidx_per_rng)
                    for v in cnts.values():
                        self.assertEqual(v, pidx_per_rng)

    def test_i_full_layer(self):
        ''' ifmap full layer. '''
        for wlkey in ['BASE', 'POOL']:
            layer = self.layers[wlkey]

            for part in self._gen_partition(wlkey=wlkey, optkey='NOINPP'):

                # Remove point from full set.
                fp_set = set(FmapRange(fp_beg=(0, 0, 0, 0),
                                       fp_end=(self.batch_size,
                                               layer.nifm,
                                               layer.hifm,
                                               layer.wifm)).range())

                for pidx in part.gen_pidx():

                    fr = partition.part_layer_ifmap_range(
                        layer, self.batch_size, part, pidx)

                    for fp in fr.range():
                        fp_set.discard(fp)

                self.assertFalse(fp_set)

    def test_i_equal_size(self):
        ''' ifmap approximately equal range size. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey,
                                                optkey='NOINPP'):

                    fr_list = []

                    for pidx in part.gen_pidx():

                        fr = partition.part_layer_ifmap_range(
                            layer, self.batch_size, part, pidx)
                        fr_list.append(fr)

                    for d in 'bnhw':
                        dsz_list = [fr.size(d) for fr in fr_list]

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

    def test_match_io_fmap_range(self):
        ''' ofmap and ifmap range match. '''
        for wlkey in self.layers:
            layer = self.layers[wlkey]

            for dnkey in self.dim_nodes:

                for part in self._gen_partition(wlkey=wlkey, dnkey=dnkey,
                                                optkey='NOINPP'):

                    for pidx in part.gen_pidx():

                        ofr = partition.part_layer_ofmap_range(
                            layer, self.batch_size, part, pidx)
                        ifr = partition.part_layer_ifmap_range(
                            layer, self.batch_size, part, pidx)

                        self.assertEqual(ofr.size('b'), ifr.size('b'))

                        if isinstance(layer, ConvLayer):
                            ol = ConvLayer(nifm=ifr.size('n'),
                                           nofm=ofr.size('n'),
                                           sofm=(ofr.size('h'), ofr.size('w')),
                                           sfil=(layer.hfil, layer.wfil),
                                           strd=(layer.htrd, layer.wtrd))
                            il = ol.input_layer()
                            self.assertEqual(il.nofm, ifr.size('n'))
                        elif isinstance(layer, LocalRegionLayer):
                            nofm_beg, nofm_end = ofr.beg_end('n')[0]
                            nifm_beg, nifm_end = ifr.beg_end('n')[0]
                            self.assertEqual(nifm_beg, max(0, \
                                    nofm_beg - layer.nreg // 2))
                            self.assertEqual(nifm_end, min(layer.nifm, \
                                    nofm_end + layer.nreg - layer.nreg // 2))

                            ol = LocalRegionLayer(nofm=ofr.size('n'),
                                                  sofm=(ofr.size('h'),
                                                        ofr.size('w')),
                                                  nreg=layer.nreg,
                                                  sreg=(layer.hreg,
                                                        layer.wreg),
                                                  strd=(layer.htrd,
                                                        layer.wtrd))
                            il = ol.input_layer()

                        self.assertEqual(il.hofm, ifr.size('h'))
                        self.assertEqual(il.wofm, ifr.size('w'))

