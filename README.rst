.. image:: https://travis-ci.org/stanford-mast/nn_dataflow.svg?branch=master
    :target: https://travis-ci.org/stanford-mast/nn_dataflow
.. image:: https://coveralls.io/repos/github/stanford-mast/nn_dataflow/badge.svg?branch=master
    :target: https://coveralls.io/github/stanford-mast/nn_dataflow?branch=master


Neural Network Dataflow Scheduling
==================================

This Python tool allows you to explore the energy-efficient dataflow scheduling
for neural networks (NNs), including array mapping, loop blocking and
reordering, and (coarse-grained) parallel processing within and across layers.

For hardware, we assume an Eyeriss-style NN accelerator [Chen16]_, i.e., a 2D
array of processing elements (PEs) with a local register file in each PE, and a
global SRAM buffer shared by all PEs. We further support a tiled architecture
with multiple nodes that can partition and process the NN computations in
parallel. Each node is an Eyeriss-style engine as above.

In software, we decouple the dataflow scheduling into three subproblems:

- Array mapping, which deals with mapping one 2D convolution computation (one
  2D ifmap convolves with one 2D filter to get one 2D ofmap) onto the hardware
  PE array. We support row stationary mapping [Chen16]_.
- Loop blocking and reordering, which decides the order between all 2D
  convolutions by blocking and reordering the nested loops. We support
  exhaustive search over all blocking and reordering schemes [Yang16]_, and
  analytical bypass solvers [Gao17]_.
- Parallel processing, which partitions the NN computations across the multiple
  tiled engines. We support both intra-layer and inter-layer parallelism. For
  intra-layer, we support batch partitioning, fmap partitioning, output
  partitioning, input partitioning, and the combination between them (hybrid)
  [Gao17]_. We also explore various dataflow optimizations including access
  forwarding and buffer sharing [Gao19]_. We use exhaustive search within each
  layer. For inter-layer, we support spatial pipelining (inter-layer
  pipelining) and temporal pipelining (time multiplexing without writing back
  intermediate data) as well as their optimized scheduling [Gao19]_. We use
  layer-wise greedy beam search across layers.

See the details in our ASPLOS'17 [Gao17]_ and ASPLOS'19 [Gao19]_ papers.

If you use this tool in your work, we kindly request that you reference our
paper(s) below, and send us a citation of your work.

- Gao et al., "TETRIS: Scalable and Efficient Neural Network Acceleration with
  3D Memory", in ASPLOS, April 2017.

- Gao et al., "TANGRAM: Optimized Coarse-Grained Dataflow for Scalable NN
  Accelerators", in ASPLOS. April 2019.


Install
-------

``nn_dataflow`` supports Python 3.6 and above.

``nn_dataflow`` can be directly used without installation if you have first
defined the environment variable ``PYTHONPATH`` to include the top directory path.
See the Usage section below for details.

``nn_dataflow`` has been registered on `PyPI
<//pypi.org/project/nn-dataflow>`_, so it can be installed through
``pip`` as::

    > pip install nn-dataflow

And ``pip`` will take care of all dependencies.

To only install ``nn_dataflow`` in local user install directory (without
``sudo``), and/or to install in editable mode, at the top directory do::

    > pip install --user -e .


Usage
-----

First, define the NN structure in ``nn_dataflow/nns``. We already defined
several popular NNs for you, including AlexNet, VGG-16, GoogLeNet, ResNet-152,
etc.

Then, use ``nn_dataflow/tools/nn_dataflow_search.py`` to search for the optimal
dataflow for the NN. For detailed options, type::

    > python ./nn_dataflow/tools/nn_dataflow_search.py -h

You can specify NN batch size and word size, PE array dimensions, number of
tile nodes, register file and global buffer capacity, and the energy cost of
all components. Note that, the energy cost of array bus should be the average
energy of transferring the data from the buffer to one PE, *not* local neighbor
transfer; the unit static energy cost should be the static energy of *all*
nodes in one clock cycle.

Other options include:

- ``-g``, ``--goal``: ``E``, ``D``, or ``ED``. the optimization goal, e(nergy),
  d(elay), or ED product.
- ``--mem-type``: ``2D`` or ``3D``. With 2D memory, memory channels are only on
  the four corners of the chip; with 3D memory, memory channels are on the top
  of all tile nodes (one per each).
- ``--bus-width``: the multicast bus bit width in the PE array for one data
  type. Set to 0 to ignore multicast overheads.
- ``--dram-bw``: ``float`` or ``inf``. Total DRAM bandwidth for all tile nodes,
  in bytes per cycle.
- ``--disable-bypass``: a combination of ``i``, ``o``, ``f``, whether to
  disallow global buffer bypass for ifmaps, ofmaps, and weights.
- ``--solve-loopblocking``: whether to use analytical bypass solvers for loop
  blocking and reordering. See [Gao17]_.
- ``--hybrid-partitioning``: whether to use hybrid partitioning in [Gao17]_.
  If not enabled, use naive partitioning, i.e., fmap partitioning for CONV
  layers, and output partitioning for FC layers.
- ``--batch-partitioning`` and ``--ifmap-partitioning``: whether the hybrid
  partitioning also explores batch and input partitioning.
- ``--enable-access-forwarding``: access forwarding, where the nodes fetch
  disjoint subsets of data and forward them to other nodes. See [Gao19]_.
- ``--enable-gbuf-sharing``: buffer sharing, where the global buffer capacity is
  shared across nodes through NoC. See [Gao19]_.
- ``--enable-save-writeback``: allow to elide the intermediate data writeback to
  memory when switching between layers if it is possible to store the entire
  data set in on-chip buffers.
- ``--interlayer-partition``: whether to use inter-layer pipelining to
  partition resources across multiple layers and process them simultaneously.
- ``--layer-pipeline-time-overhead``, ``--layer-pipeline-max-degree``:
  constrain the configuration space of inter-layer pipelining, by specifying
  the maximum execution time overhead, or the maximum pipelining degree.
- ``--disable-interlayer-opt``: disable optimizations and only allow basic
  inter-layer pipelining.


Code Structure
--------------

- ``nn_dataflow``
    - ``core``
        - Top-level dataflow exploration: ``nn_dataflow``,
          ``nn_dataflow_scheme``.
        - Layer scheduling: ``scheduling``.
        - Array mapping: ``map_strategy``.
        - Loop blocking and reordering: ``loop_blocking``,
          ``loop_blocking_scheme``, ``loop_blocking_solver``.
        - Intra-layer partitioning: ``partition``, ``partition_scheme``,
          ``buf_shr_scheme``.
        - Inter-layer pipelining: ``inter_layer_pipeline``,
          ``pipeline_segment``.
        - Network and layer: ``network``, ``layer``.
    - ``nns``: example NN definitions.
    - ``tests``: unit tests.
    - ``tools``: executables.


Verification and Testing
------------------------

To verify the tool against the Eyeriss result [Chen16]_, see
``nn_dataflow/tests/dataflow_test/test_nn_dataflow.py``.

To run (unit) tests, do one of the following::

    > python -m unittest discover

    > python -m pytest

    > pytest

To check code coverage with ``pytest-cov`` plug-in::

    > pytest --cov=nn_dataflow


Copyright & License
-------------------

``nn_dataflow`` is free software; you can redistribute it and/or modify it
under the terms of the `BSD License <LICENSE>`__ as published by the Open
Source Initiative, revised version.

``nn_dataflow`` was originally written by Mingyu Gao at Stanford University,
and per Stanford University policy, the copyright of this original code remains
with the Board of Trustees of Leland Stanford Junior University.


References
----------

.. [Gao19] Gao, Yang, Pu, Horowitz, and Kozyrakis, `TANGRAM: Optimized
  Coarse-Grained Dataflow for Scalable NN Accelerators
  <//dl.acm.org/citation.cfm?id=3297858.3304014>`__, in ASPLOS. April, 2019.

.. [Gao17] Gao, Pu, Yang, Horowitz, and Kozyrakis, `TETRIS: Scalable and
  Efficient Neural Network Acceleration with 3D Memory
  <//dl.acm.org/citation.cfm?id=3037697.3037702>`__, in ASPLOS. April, 2017.

.. [Chen16] Chen, Emer, and Sze, `Eyeriss: A Spatial Architecture for
  Energy-Efficient Dataflow for Convolutional Neural Networks
  <//dl.acm.org/citation.cfm?id=3001177>`__, in ISCA. June, 2016.

.. [Yang16] Yang, Pu, Rister, Bhagdikar, Richardson, Kvatinsky,
  Ragan-Kelley, Pedram, and Horowitz, `A Systematic Approach to Blocking
  Convolutional Neural Networks <//arxiv.org/abs/1606.04209>`__, arXiv
  preprint, 2016.

