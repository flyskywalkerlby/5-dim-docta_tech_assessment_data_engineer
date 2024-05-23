import json

from tqdm import tqdm

with open("data/test.json", "r") as f:
    data = json.load(f)


sample_data = []

for i, row in enumerate(tqdm(data)):
    idx = row["id"]
    text = row["input"]

    sample_data.append(dict(
        id=idx,
        input=text
    ))

    if i == 10:
        break

with open("data/test_10.json", "w") as f:
    json.dump(sample_data, f)
