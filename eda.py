from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("AdaptLLM/finance-tasks", "Headline")["test"]
print(len(dataset))

"""
num of rows: 20547
columns: dict_keys(['gold_index', 'class_id', 'input', 'id', 'options'])
"""

gold_indexes = set()
class_ids = set()
ids = set()
options = set()

for i, row in enumerate(tqdm(dataset)):
    gold_indexes.add(row["gold_index"])
    class_ids.add(row["class_id"])
    ids.add(row["id"])
    options.update(row["options"])

    # print(row["options"])

    text = row["input"]
    print(i)
    print(text)

    if i == 10:
        break

    # if len(ids) % 1000 == 0:  # Optionally print progress every 1000 unique ids found
    #     print(f"Processed {len(ids)} unique ids.")

print("Collection complete.")
print(f"Unique gold indexes: {len(gold_indexes)}", gold_indexes)
print(f"Unique class IDs: {len(class_ids)}", class_ids)
print(f"Unique IDs: {len(ids)}")
print(f"Unique options: {len(options)}", options)
