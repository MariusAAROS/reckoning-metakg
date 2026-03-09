from datasets import load_dataset
from collections import defaultdict
import json, os

os.makedirs("data/proofwriter", exist_ok=True)

# ProofWriter depth configs: depth-0 to depth-5 (depth-4 doesn't exist)
DEPTHS = ["depth-0", "depth-1", "depth-2", "depth-3", "depth-5"]

# HuggingFace allenai/proofwriter is flat (one row per question).
# ProofWriterDataReader expects grouped records: lists of questions/answers per theory.
# We group by theory ID (the part of `id` before the last `_Q` suffix).
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def group_by_theory(split_data, depth_prefix):
    """Group flat QA rows by their shared theory into the schema expected by ProofWriterDataReader."""
    groups = {}
    for row in split_data:
        # id format: "AttNeg-D0-289_Q1" → theory key is "AttNeg-D0-289"
        raw_id = row["id"]
        theory_key = raw_id.rsplit("_", 1)[0] if "_Q" in raw_id else raw_id
        unique_key = f"{depth_prefix}_{theory_key}"

        if unique_key not in groups:
            # Facts are one per line in the theory string
            facts = [s.strip() for s in row["theory"].split("\n") if s.strip()]
            groups[unique_key] = {"facts": facts, "questions": [], "answers": []}

        groups[unique_key]["questions"].append(row["question"])
        # answers must be "true" / "false" / "unknown" to match answer_map in ProofWriterDataReader
        groups[unique_key]["answers"].append(row["answer"].lower())

    return groups


# Collect grouped records per output split across all depths
all_groups = {out_split: {} for out_split in SPLIT_MAP.values()}

for depth in DEPTHS:
    ds = load_dataset("tasksource/proofwriter")
    for hf_split, out_split in SPLIT_MAP.items():
        if hf_split not in ds:
            continue
        groups = group_by_theory(ds[hf_split], depth_prefix=depth)
        all_groups[out_split].update(groups)

# Write output files
for out_split, groups in all_groups.items():
    with open(f"data/proofwriter/{out_split}.jsonl", "w") as f:
        for i, (_, data) in enumerate(groups.items()):
            record = {
                "guid": f"{out_split}-{i}",
                "facts": data["facts"],
                "questions": data["questions"],
                "answers": data["answers"],
            }
            json.dump(record, f)
            f.write("\n")
