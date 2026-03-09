from datasets import load_dataset, concatenate_datasets
import json, os

os.makedirs("data/babilong", exist_ok=True)

N_SUBSETS = 10
RATIO_TRAIN = 0.95
RATIO_VAL = (1-RATIO_TRAIN) * 0.5
RATIO_TEST = (1-RATIO_TRAIN) * 0.5

subsets = [f"qa{i+1}" for i in range(N_SUBSETS)]
included_splits = ["0k"]

# Collect all data per split across subsets
all_data = {s: [] for s in included_splits}
for subset in subsets:
    ds = load_dataset("RMT-team/babilong", subset)
    for split_name, split_data in ds.items():
        if split_name in included_splits:
            all_data[split_name].append(split_data)

# Concatenate across subsets and write to files
for split_name, datasets in all_data.items():
    if not datasets:
        continue
    combined = concatenate_datasets(datasets).shuffle(seed=42)

    split_size = len(combined)
    train_end = int(split_size * RATIO_TRAIN)
    val_end = train_end + int(split_size * RATIO_VAL)
    test_end = val_end + int(split_size * RATIO_TEST)

    for i, row in enumerate(combined):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        elif i < test_end:
            split = "test"
        else:
            break
        record = {
            "guid": f"{split}-{i}",
            "facts": [s.strip() + "." for s in row["input"].split(". ") if s.strip()],
            "question": row["question"],
            "answer": row["target"],
            "support": row.get("input", "")
        }
        with open(f"data/babilong/{split}.jsonl", "a") as f:
            json.dump(record, f)
            f.write("\n")