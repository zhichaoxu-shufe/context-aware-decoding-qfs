import os
import sys
import json
import time

from datasets import load_dataset

dataset = load_dataset("EdinburghNLP/xsum")

input_dir = "/uusoc/exports/scratch/brutusxu/decoding/datasets/"

if not os.path.isdir(input_dir, "xsum"):
    os.mkdir(os.path.join(input_dir, "xsum"))

with open(os.path.join(input_dir, "xsum/train.jsonl"), "w") as fout:
    for i, row in enumerate(dataset["train"]):
        json.dump(row, fout)
        fout.write("\n")

with open(os.path.join(input_dir, "xsum/validation.jsonl"), "w") as fout:
    for i, row in enumerate(dataset["validation"]):
        json.dump(row, fout)
        fout.write("\n")

with open(os.path.join(input_dir, "xsum/test.jsonl"), "w") as fout:
    for i, row in enumerate(dataset["test"]):
        json.dump(row, fout)
        fout.write("\n")


if not os.path.join(input_dir, "cnn_dailymail"):
    os.mkdir(os.path.join(input_dir, "cnn_dailymail"))

dataset = load_dataset("cnn_dailymail", "1.0.0")
with open(os.path.join(input_dir, "cnn_dailymail/train.jsonl"), "w") as fout:
    for i, row in enumerate(dataset["train"]):
        row_ = {"document": row["article"], "summary": row["highlights"]}
        json.dump(row_, fout)
        fout.write("\n")

with open(os.path.join(input_dir, "cnn_dailymail/validation.jsonl"), "w") as fout:
    for i, row in enumerate(dataset["validation"]):
        row_ = {"document": row["article"], "summary": row["highlights"]}
        json.dump(row_, fout)
        fout.write("\n")

with open(os.path.join(input_dir, "cnn_dailymail/test.jsonl"), "w") as fout:
    for i, row in enumerate(dataset["test"]):
        row_ = {"document": row["article"], "summary": row["highlights"]}
        json.dump(row_, fout)
        fout.write("\n")

