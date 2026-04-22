# source venv/bin/activate
# pip install matplotlib
# results from nn_dataflow have to be put in a json file
# $ [command to run nn_dataflow] > [filename.json]
# python [filename w/python script].py [filename.json] <- can cause errors
# explorer.exe [saved filename].png <- this works

import matplotlib.pyplot as plt
import json
import sys
import os
import re

fileEndExt = ".png"

# check the # of arguments and terminate the script if error present exit code 1.
if len(sys.argv) < 2:
  print("Error: not enough arguments\n")
  print("Example: python display_results_energy.py [filenname].json\n")
  sys.exit(1)

# check if the 2nd argument file exists and it is a .json file
fileName = sys.argv[1]

if not os.path.exists(fileName):
    print(f"Error: file '{fileName}' not found")
    sys.exit(1)

if not fileName.endswith(".json"):
    print(f"Warning: '{fileName}' is not a .json file")
    sys.exit(1)

# opening provided .json file and validate .json keys used for this file
try:
    with open(fileName) as file:
        fileData = json.load(file)
except json.JSONDecodeError as e:
    print(f"Error: could not parse JSON - {e}")
    sys.exit(1)

required_keys = ["total_cost", "total_time", "active_node_pes", "schedules"]
for key in required_keys:
    if key not in fileData:
        print(f"Error: missing expected key '{key}' in JSON")
        sys.exit(1)

# Read the data from the .json file
totalEnergy = fileData["total_cost"]
totalLatency = fileData["total_time"]
pesInLayer = fileData["active_node_pes"]

# display_results_basic.py reads the single "cost" field per layer,
# instead this script pulls the three energy sub-components: cost_op (compute),
# cost_access (memory hierarchy), and cost_noc (network-on-chip) so we can see
# where each layer's total energy is actually going.
schedules = fileData["schedules"]
layerOpDict = {}
layerAccessDict = {}
layerNocDict = {}
for layer, sched in schedules.items():
    if sched and isinstance(sched[0], dict):
        layerOpDict[layer] = sched[0].get("cost_op", 0)
        layerAccessDict[layer] = sched[0].get("cost_access", 0)
        layerNocDict[layer] = sched[0].get("cost_noc", 0)

layers = list(pesInLayer.keys())

opVals = [layerOpDict.get(l, 0)     for l in layers]
accessVals = [layerAccessDict.get(l, 0) for l in layers]
nocVals = [layerNocDict.get(l, 0)    for l in layers]

# plot (AI assisted)
# all three values are added into one bar since they are the same unit of measurement
plt.style.use("ggplot")
fig, ax = plt.subplots(figsize=(20, 10), dpi=150)

xvals = range(len(layers))

# stacked bars: compute (cost_op) on bottom, memory access in the middle, NoC on top
barOp     = ax.bar(xvals, opVals,     color="steelblue",
                   edgecolor="white", linewidth=0.6, label="Compute (cost_op)")
barAccess = ax.bar(xvals, accessVals, bottom=opVals,
                   color="darkorange", edgecolor="white", linewidth=0.6,
                   label="Memory Access (cost_access)")
accessTop = [o + a for o, a in zip(opVals, accessVals)]
barNoc    = ax.bar(xvals, nocVals, bottom=accessTop,
                   color="mediumseagreen", edgecolor="white", linewidth=0.6,
                   label="Network-on-Chip (cost_noc)")

ax.set_xticks(list(xvals))
ax.set_xticklabels(layers, rotation=90, ha="center", fontsize=7)
ax.tick_params(axis="y", labelsize=8)
ax.set_ylabel("Energy Cost", fontsize=11, labelpad=8)
ax.set_title("Per-Layer Energy Breakdown: Compute vs Memory Access vs NoC",
             fontsize=12, fontweight="bold", pad=10)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=9)

layerTotals = [o + a + n for o, a, n in zip(opVals, accessVals, nocVals)]
maxTotal = max(layerTotals) if layerTotals else 0
# added a little headroom at the top because during some testing the graph went off of the screen
if maxTotal > 0:
    ax.set_ylim(0, maxTotal * 1.08) # this .08

totalOp = sum(opVals)
totalAccess = sum(accessVals)
totalNoc = sum(nocVals)

fig.suptitle(
    f"Total Energy: {totalEnergy:.2f}  |  "
    f"Compute: {totalOp:.2f}  |  Access: {totalAccess:.2f}  |  NoC: {totalNoc:.2f}",
    fontsize=13, fontweight="bold", y=1.02
)

plt.tight_layout()
# End of AI Assistance

# Obtain output.png filename from the user while checking for formatting, characters, and existing files
while True:
    print("Please enter a filename for the created .png file.")
    userFileInput = input().strip()

    if not userFileInput:
        print("Error: filename cannot be empty. Please try again")
        continue

    if not re.match(r'^[\w\-. ]+$', userFileInput):
        print("Error: filename contains invalid characters")
        print("Valid characters: letters, numbers, hyphens, underscores, and spaces")
        continue

    fullFileName = userFileInput + fileEndExt

    if os.path.exists(fullFileName):
        print(f"'{fullFileName}' already exists. Overwrite existing file? (y/n)")
        if input().strip().lower() != 'y':
            continue
    break

plt.savefig(fullFileName)
try:
    plt.show()
except:
    pass

print(f"Saved: {fullFileName}")
print(f"To open on Windows run: explorer.exe {fullFileName}")

# if UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown plt.show()
# then explorer.exe [filename].png
