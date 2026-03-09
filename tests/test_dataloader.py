import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from meta_kg.dataset import MetaKnowledgeDataset
from meta_kg.module import get_features

args = OmegaConf.create({
    "dataset": "babilong", "dataset_type": "babilong",
    "load_order": "in", "no_facts": False, "random_facts": False,
    "no_question": False, "max_eval_data": 5, "baseline": False,
    "multi_task": False, "do_eval": False,
})
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
ds = MetaKnowledgeDataset(args, tokenizer, "data", "test", is_training=False)
sample = ds[0]

print("=== INNER LOOP TRAIN TARGETS ===")
tok = tokenizer
for ids, lbl in zip(sample["train_input_ids"], sample["train_labels"]):
    print("Prompt+completion:", tok.decode(ids))
    valid = lbl[lbl != -100]
    print("Labels only:", tok.decode(valid))
    print("---")

print("=== OUTER LOOP DEV TARGET ===")
for ids, lbl in zip(sample["input_ids"], sample["labels"]):
    print("Prompt+completion:", tok.decode(ids))
    valid = lbl[lbl != -100]
    print("Labels only:", tok.decode(valid))
    print("---")

print("=== PRINT OUT ===")
print(sample["print_out"])