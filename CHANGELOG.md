List of major changes and improvements
======================================

## [Unreleased]

### Added

- Explorers and solvers.

  - Performance improvements.
    - Add loop blocking scheme cache in Scheduling.
    - Use a single Scheduling instance for all same layers.

- Software engineering

  - Class instance used as dict key.
    - Add value-based equality and hash to Layer.
    - Add value-based equality and hash to PartitionScheme.

  - Add version number to output json dump.


### Changed

- Explorers and solvers:

  - Better formatting of verification results against Eyeriss.

- Software engineering

  - Replace numpy for better performance.

  - Move multiprocessing into loop blocking exploration for better performance
    scaling.


## [v1.0 -- v1.1] -- 2017-05-04

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


## [init -- v1.0] -- 2017-01-21

### Added

- Workload models:

  - Two layer types: convolutional layer and fully-connected layer.

  - Supported neural neworks: AlexNet, VGG, VGG19, ZFNet, ResNet152.

  - Supported data categories: ifmaps, ofmaps, and weights.

- Hardware models:

  - 2D Network-on-Chip (NoC) on the PE array (node) level.

  - 2D PE array on the PE level.

  - Memory hierarchy:
    - regf: register file in a PE.
    - itcn: interconnect between PEs in an array.
    - gbuf: global buffer of an array.
    - dram: main memory.

  - Cost (energy) of computation operations (MAC, etc.), memory hierarchy
    accesses, NoC hop traversals, and static leakage.

- Software models:

  - Eyeriss Row-Stationary mapping to PE array (Chen et al., ISCA 2016).

  - Loop blocking schemes over ifmap channel, ofmap channel, and batch loops.
    - Loop reordering: exchange loop order.
    - Loop blocking: split a loop into multiple ones.

  - Partition schemes to split the workload of a layer to different nodes.
    - Fmap partitioning: partition height/width of a fmap.
    - Output partitioning: partition different fmaps (channels).

- Explorers and solvers:

  - Per-layer schedule exploration.

  - Exhaustive search loop blocking schemes and partitioning schemes.

  - Analytically solve bypass loop ordering (Gao et al., ASPLOS 2017).

  - Naive partitioning scheme (Kim et al., ISCA 2016).

- Software engineering

  - Support multi-process parallel processing.

