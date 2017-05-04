List of major changes and improvements
======================================

## [Unreleased]

### Added

- Workload models.

  - `Network`: a DAG of layers, rather than a linear pipeline.

  - New layer types: pooling layer (local region layer).

  - Enforce layer chaining has matched data size.

  - New neural network: GoogLeNet.

- Hardware models.

  - `NodeRegion`.
    - Used to denote memory regions, i.e., relative positions and sizes of
      memories to the computation node NoC.
    - Support 2D memories, which are on the edges of the chip.

- Software models.

  - `FmapPosition` and `FmapRange`: a position and a range in batched fmaps.
    - `FmapRangeMap`: efficient map structure of `FmapPosition` type.

  - `DataLayout`: describes the layer i/ofmap data layout.
    - Use a `FmapRangeMap` to map each data element to the stored node.

  - Partition schemes.
    - Batch partitioning: partition input data within a batch.

- Explorers and solvers.

  - `SchedulingResultDict`: store layer scheduling results of a network.

  - More checks to enforce the schedules have the correct number of operations
    as the given workloads.


### Changed

- Workload models.

  - Update all network structures to include pooling layers.

- Software models.

  - Allow different partitioning factor along height and width of a fmap, i.e.,
    allow different height and width sizes of the partitioned fmap.

- Explorers and solvers.

  - `NNDataflow`: new top-level class.

- Software engineering:

  - Significant code refactoring to improve modularity.
    - More classes, e.g., `MapStrategy`, `LoopBlockingScheme`, `Scheduling`.

  - Code style lint.

  - Update option names to be more uniform.

  - Standardize class stringify.


### Deprecated

- Option `--hybrid_partition2d`, now is `--hybrid_partition`.


### Fixed

- Use of `map()` function in `PhyDim2`.

- Name of `namedtuple` sublasses.

- Structure configuration of ResNet152.

