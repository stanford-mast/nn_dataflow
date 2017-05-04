List of major changes and improvements
======================================

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

