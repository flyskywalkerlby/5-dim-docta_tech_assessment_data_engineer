import re
import time
import json

from tqdm import tqdm

with open("data/test.json", "r") as f:
    data = json.load(f)


"""
num of rows: 20547
columns: dict_keys(['gold_index', 'class_id', 'input', 'id', 'options'])
"""


start_t = time.time()

pairs = {}
formatted_data = {}

suc = 0
fail = 0

for i, row in enumerate(tqdm(data)):

    idx = row["id"]
    text = row["input"]

    # print(i)

    for pair_i, pair in enumerate(text.split("\n\n")):
        _id = f"{idx}-{pair_i}"

        # print("==")
        # print(pair)
        # print("==")

        answer_patterns = (
            r'Answer: (Yes|No)$',
            r'(Yes|No)$'
        )

        for pt in answer_patterns:
            answer_match = re.search(pt, pair, re.MULTILINE)
            if answer_match:
                answer = answer_match.group(0)
                question = re.sub(pt, '', pair).strip()
                break
            suc += 1
        else:
            fail += 1
            answer = ""
            question = pair

        if answer:
            formatted_data[_id] = {
                'id': _id,
                'question': question,
                'answer': answer
            }

        pairs[_id] = pair

    # if i == 100:
    #     break

# print(formatted_data)
print(f"success: {suc}, fail: {fail}")

with open("data/formatted_answer_extracted.json", "w") as f:
    json.dump(formatted_data, f, indent=4)

with open("data/pairs.json", "w") as f:
    json.dump(pairs, f, indent=4)


print(f"Takes {time.time() - start_t}s")
