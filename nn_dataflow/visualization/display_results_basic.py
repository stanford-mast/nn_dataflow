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
  print("Example: python display_results_basic.py [filenname].json\n")
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
    print(f"Error: could not parse JSON — {e}")
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

# Extract per-layer cost and time directly from schedules
schedules = fileData["schedules"]
layerCostDict = {}
layerTimeDict = {}
for layer, sched in schedules.items():
    if sched and isinstance(sched[0], dict):
        layerCostDict[layer] = sched[0].get("cost", 0)
        layerTimeDict[layer] = sched[0].get("time", 0)

layers = list(pesInLayer.keys())
pes = list(pesInLayer.values())

# plot (AI assisted)
plt.style.use("ggplot")
fig, axes = plt.subplots(3, 1, figsize=(20, 15), dpi=150)

def styled_bar(ax, xvals, yvals, color, ylabel, title):
    ax.bar(xvals, yvals, color=color, edgecolor="white", linewidth=0.8)
    ax.set_xticks(xvals)
    ax.set_xticklabels(layers, rotation=75, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

xvals = range(len(layers))

layerCostVals = [layerCostDict.get(l, 0) for l in layers]
layerTimeVals = [layerTimeDict.get(l, 0) for l in layers]

styled_bar(axes[0], xvals, layerCostVals, "steelblue",     "Energy Cost",     "Energy per Layer")
styled_bar(axes[1], xvals, layerTimeVals, "darkorange",    "Latency (cycles)", "Latency per Layer")
styled_bar(axes[2], xvals, pes,           "mediumseagreen", "Active PEs",      "Active PEs per Layer")

fig.suptitle(
    f"Total Energy: {totalEnergy:.2f}  |  Total Latency: {totalLatency:.2f}",
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