# nn_dataflow Contributions

This README covers three contributions added to a fork from https://github.com/rpiza12/nn_dataflow.git:

1. `visualization/display_results_basic.py`: a -goal visualization script for nn_dataflow JSON output
2. `visualization/display_results_energy.py`: a per-layer energy breakdown visualization script for nn_dataflow JSON output
3. `nn_dataflow/nns/mobilenet_v2.py`: a MobileNetV2 example network definition

## 1. Visualization Script (Basic)
**File:** `visualization/display_results_basic.py`

### Overview
Uses a JSON output file that is produced by `nn_dataflow_search.py` and creates a three panel bar chart that shows layer-wise hardware metrics for any NN and optimization goal: 
- **Energy per layer** cost values from the optimal schedule found for each layer
- **Latency per layer** time in cycles from the optimal schedule found for each layer  
- **Active PEs per layer** number of processing elements active during each layer

### Setup
0) Create a virutal environment that will contain separate python packages and installations
```bash
source venv/bin/activate
pip install matplotlib
```

### Usage
1) Run the nn_dataflow search and redirect the output to a JSON file, example:

```bash
python3 -m nn_dataflow.tools.nn_dataflow_search resnet50 \
--batch 16 \
--nodes 1 1 \
--array 16 16 \
--regf 512 \
--gbuf 128000 \
> resnet50_baseline.json
```

2) run the visualization script:

```bash
python visualization/display_results_basic.py resnet50_baseline.json
```

3) The script will prompt the User to enter a filename. Saving the chart as a `.png` file in the current directory.

```
Please enter a filename for the created .png file.
chart1
Saved: chart1.png
To open on Windows run: explorer.exe my_chart.png
```

**Note:** If `plt.show()` those an error use `explorer.exe filename.png` to open the saved file on Windows.

### Script Input validation
The script validates the following inputs:
- Checks that a JSON filename argument was provided
- Verifies the file exists and has a `.json` extension
- Catches JSON missing required keys (`total_cost`, `total_time`, `active_node_pes`)
- Validates the output filename for empty input, invalid characters, and existing file conflicts

## 2. Visualization Script (Advanced)
**File:** `visualization/display_results_energy.py`

### Overview
Uses a JSON output file that is produced by `nn_dataflow_search.py` and creates a single stacked bar chart that shows where each layer's energy is actually going. Unlike `display_results_basic.py`, which reads the single `cost` field per layer, this script pulls the three energy sub-components from `schedules[layer][0]`:
- **Compute (`cost_op`)** energy spent doing the MAC operations themselves
- **Memory Access (`cost_access`)** energy spent moving data through the memory hierarchy
- **Network-on-Chip (`cost_noc`)** energy spent on inter-node data movement

Each layer appears as one stacked bar, with the three components stacked bottom-to-top so the full bar height equals that layer's total energy cost.

### Setup
0) Create a virutal environment that will contain separate python packages and installations
```bash
source venv/bin/activate
pip install matplotlib
```

### Usage
1) Run the nn_dataflow search and redirect the output to a JSON file, example:

```bash
python3 -m nn_dataflow.tools.nn_dataflow_search resnet50 \
--batch 16 \
--nodes 1 1 \
--array 16 16 \
--regf 512 \
--gbuf 128000 \
> resnet50_baseline.json
```

2) run the visualization script:

```bash
python visualization/display_results_energy.py resnet50_baseline.json
```

3) The script will prompt the User to enter a filename. Saving the chart as a `.png` file in the current directory.

```
Please enter a filename for the created .png file.
energychart1
Saved: energychart1.png
To open on Windows run: explorer.exe my_chart.png
```

**Note:** If `plt.show()` those an error use `explorer.exe filename.png` to open the saved file on Windows.

### Script Input validation
The script validates the following inputs:
- Checks that a JSON filename argument was provided
- Verifies the file exists and has a `.json` extension
- Catches JSON missing required keys (`total_cost`, `total_time`, `active_node_pes`, `schedules`)
- Validates the output filename for empty input, invalid characters, and existing file conflicts

## 3. Neural Network Definition: MobileNetV2
**File:** `./nn_dataflow/nns/mobilenet_v2.py`

### What it does
Adds MobileNetV2 as a supported network in nn_dataflow. Following the same conventions as existing NN definitions, AlexNet, VGG-16, ResNet-50, etc, in `./nn_dataflow/nns/`. MobileNetV2 can be used directly with `nn_dataflow_search.py`.

### Architecture
MobileNetV2 is a CNN built around inverted residual blocks. With each block consisting of three operations:
1. **1×1 pointwise expand conv** expands the channel count by an expansion factor, `t`.
2. **3×3 depthwise conv** filters features channel-by-channel, modeled as `ConvLayer` with `in_channels == out_channels`. The stride `s` from table 2 in https://arxiv.org/pdf/1801.04381.
3. **1×1 pointwise project conv** projects back down to the output channel count, `c`.

The notation follows Table 2 of the paper:
- `t` — expansion factor applied to input channels before the depthwise conv
- `c` — number of output channels for the block
- `n` — number of times the block is repeated in sequence
- `s` — stride applied to the depthwise conv of the first block in the sequence only; all subsequent repeated blocks use stride 1

Blocks are named to reflect the table's grouping. As an example `b2_1` and `b2_2` are the two repetitions of the `t = 6, c = 24, n = 2` row in Table 2.

**Note on depthwise convolutions:** The nn_dataflow framework does not have a dedicated depthwise layer type. Depthwise convolutions are modeled as `ConvLayer` with equal input and output channels, consistent with how other nn_dataflow network definitions handle similar approximations.

**Note on batch normalization:** Batch normalization layers are ignored. nn_dataflow has no `BatchNormLayer` type and norm layers add no compute or memory cost relevant to dataflow scheduling.

### Modeled after the paper
The implementation followed Table 2 of the MobileNetV2 paper. 
The full architecture against Table 2 of the paper:

| Paper row           | t | c  | n | s | Implementation |
|---------------------|---|----|---|---|----------------|
| 224²×3 conv2d       | - | 32 | 1 | 2 | `conv1` |
| 112²×32 bottleneck  | 1 | 16 | 1 | 1 | `b1` |
| 112²×16 bottleneck  | 6 | 24 | 2 | 2 | `b2_1`, `b2_2` |
| 56²×24 bottleneck   | 6 | 32 | 3 | 2 | `b3_1`, `b3_2`, `b3_3` |
| 28²×32 bottleneck   | 6 | 64 | 4 | 2 | `b4_1`, `b4_2`, `b4_3`, `b4_4` |
| 14²×64 bottleneck   | 6 | 96 | 3 | 1 | `b5_1`, `b5_2`, `b5_3` |
| 14²×96 bottleneck   | 6 | 160 | 3 | 2 | `b6_1`, `b6_2`, `b6_3` |
| 7²×160 bottleneck   | 6 | 320 | 1 | 1 | `b7` |
| 7²×320 conv2d 1×1   | - | 1280 | 1 | 1 | `conv_last` |
| 7²×1280 avgpool 7×7 | - | - | 1 | - | `pool_last` |
| 1×1×1280 conv2d 1×1 | - | k | - | - | `fc` |

### Run the dataflow search

```bash
python ./nn_dataflow/tools/nn_dataflow_search.py mobilenet_v2 \
--batch 1 \
--word 16 \
--array 16 16 \
--nodes 1 1 \
--regf 64 \
--gbuf 131072 \
> mobilenet_v2_results.json
```

### Reference

Sandler, Howard, Zhu, Zhmoginov, and Chen. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018. https://arxiv.org/abs/1801.04381
With further help from https://www.geeksforgeeks.org/computer-vision/what-is-mobilenet-v2/