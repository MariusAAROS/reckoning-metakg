from datasets import load_dataset
import json, os

os.makedirs("data/babilong", exist_ok=True)

ds = load_dataset("RMT-team/babilong", "qa1")  # adjust subset as needed

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

included_splits = ["0k"]
for split_name, split_data in ds.items():
    if not split_name in included_splits:
        continue

    split_size = len(split_data)
    train_end = int(split_size * TRAIN_SPLIT)
    val_end = train_end + int(split_size * VAL_SPLIT)
    test_end = val_end + int(split_size * TEST_SPLIT)

    split_data = split_data.shuffle(seed=42)
    for i, row in enumerate(split_data):
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
            "facts": [s.strip() for s in row["input"].split("\n") if s.strip()],
            "question": row["question"],
            "answer": row["target"],
            "support": row.get("input", "")
        }
        with open(f"data/babilong/{split}.jsonl", "a") as f:
            json.dump(record, f)
            f.write("\n")