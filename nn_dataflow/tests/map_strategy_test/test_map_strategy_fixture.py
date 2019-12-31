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

import unittest
from collections import OrderedDict

from nn_dataflow.core import ConvLayer, FCLayer, LocalRegionLayer, PoolingLayer
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource

class TestMapStrategyFixture(unittest.TestCase):
    ''' Base fixture class for MapStrategy tests. '''

    def setUp(self):

        # AlexNet.
        self.convlayers = OrderedDict()
        self.convlayers['conv1'] = ConvLayer(3, 96, 55, 11, 4)
        self.convlayers['conv2'] = ConvLayer(48, 256, 27, 5)
        self.convlayers['conv3'] = ConvLayer(256, 384, 13, 3)
        self.convlayers['conv4'] = ConvLayer(192, 384, 13, 3)
        self.convlayers['conv5'] = ConvLayer(192, 256, 13, 3)
        self.fclayers = {}
        self.fclayers['fc1'] = FCLayer(256, 4096, 6)
        self.fclayers['fc2'] = FCLayer(4096, 4096)
        self.fclayers['fc3'] = FCLayer(4096, 1000)

        # LocalRegionLayer.
        self.lrlayers = {}
        self.lrlayers['pool1'] = PoolingLayer(64, 7, 2)
        self.lrlayers['pool2'] = PoolingLayer(29, 13, 3)
        self.lrlayers['pool3'] = PoolingLayer(32, 7, 2, strd=3)
        self.lrlayers['lr1'] = LocalRegionLayer(32, 7, nreg=5, sreg=1)
        self.lrlayers['lr2'] = LocalRegionLayer(32, 7, nreg=5, sreg=1, strd=2)

        # Fake layers.
        self.fake_layers = {}
        # With irregular nifm/nofm.
        self.fake_layers['IRR'] = ConvLayer(255, 383, 13, 3)
        # With small numbers of fmaps.
        self.fake_layers['SM'] = ConvLayer(5, 3, 13, 3)
        # With large FIL height.
        self.fake_layers['LGFIL'] = ConvLayer(64, 64, 13, 22)

        # Resource.
        self.resource = {}
        proc_region = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                 type=NodeRegion.PROC)
        data_region = NodeRegion(origin=PhyDim2(0, 0), dim=PhyDim2(1, 1),
                                 type=NodeRegion.DRAM)
        # Eyeriss, ISSCC'16, JSSC'17.
        self.resource['BASE'] = Resource(
            proc_region=proc_region, dram_region=data_region,
            src_data_region=data_region, dst_data_region=data_region,
            dim_array=PhyDim2(12, 14), size_gbuf=108*1024, size_regf=520,
            array_bus_width=float('inf'), dram_bandwidth=float('inf'),
            no_time_mux=False)

