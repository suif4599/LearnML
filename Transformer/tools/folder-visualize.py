import argparse
import os
import json
from math import inf

parser = argparse.ArgumentParser(description='Visualize all models in a folder')
parser.add_argument("folder", type=str, help='Folder containing models')
parser.add_argument("--no-color", action="store_true", help='Disable color output')
parser.add_argument("--indent", type=int, default=4, help='Indentation')
args = parser.parse_args()

if not os.path.exists(args.folder):
    print(f"Folder {args.folder} does not exist")
    exit()

def parse_folder(folder, indent=4, color=True):
    res = []
    model_name = "".join(os.path.basename(folder).split("_")[:-1])
    res.append(f"Model: {model_name}")
    if color:
        res[0] = "\033[1;32m" + res[0] + "\033[0m"
    if not os.path.exists(os.path.join(folder, "metadata.json")):
        print(f"Model {model_name} is incomplete")
        return
    res.append("\tMetadata:")
    if color:
        res[-1] = "\033[1;34m" + res[-1] + "\033[0m"
    with open(os.path.join(folder, "metadata.json"), "r") as f:
        metadata = json.load(f)
        for k, v in metadata.items():
            res.append(f"\t\t{k}: {v}")
    res.append("\tTrain info:")
    if color:
        res[-1] = "\033[1;34m" + res[-1] + "\033[0m"
    with open(os.path.join(folder, "train_log.json"), "r") as f:
        train_log = json.load(f)
    n_epoch = sum(k.startswith("epoch") for v in train_log.values() for k in v)
    res[0] += f" ({n_epoch} epochs)"
    min_test_loss = min(v["test_loss"] for v in train_log.values() for k, v in v.items() if k.startswith("epoch"))
    res.append(f"\t\tMinimum test loss: {min_test_loss:.2f}")
    current_loss = list(v["test_loss"] for v in train_log.values() for k, v in v.items() if k.startswith("epoch"))[-1]
    res.append(f"\t\tCurrent test loss: {current_loss:.2f}")
    print("\n".join(res).replace("\t", " " * indent))


for f in sorted(filter(lambda folder: os.path.isdir(folder), os.listdir(args.folder)), key=lambda folder: os.path.basename(folder).split("_")[-1]):
    parse_folder(os.path.join(args.folder, f), args.indent, not args.no_color)

