import json
import time

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
# model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, load_in_8bit=True)
model = model.eval()


def infer(text):

    # text: question and headline

    input_text = (f"Extract the headline and question from the text without altering the content: {text}. "
                  f"Return the result in a JSON dictionary format with keys \"headline\" and \"question\". "
                  f"##Answer: {{\"headline\": \"\", \"question\": \"\"}}")

    response, history = model.chat(tokenizer, input_text, history=[])

    try:
        qh = json.loads(response)
    except json.JSONDecodeError:
        print(response)
        qh = None

    return qh


start_t = time.time()


with open("data/formatted_answer_extracted.json", "r") as f:
    data = json.load(f)


formatted_data = {}


for i, (_id, pair) in enumerate(tqdm(data.items())):

    question = pair["question"]

    qh = infer(question)

    if qh:
        pair.update(
            question=qh["question"],
            headline=qh["headline"]
        )

        formatted_data[_id] = pair

    if i == 50:
        break


print(len(formatted_data), len(data))

with open("data/formatted_qh_extracted.json", "w") as f:
    json.dump(formatted_data, f, indent=4)

print(f"Takes {time.time() - start_t}s")
