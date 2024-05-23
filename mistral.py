import json
import time

from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
print("Loaded")
# quit()


def infer(pair_str):
    messages = [
        {
            "role": "user",
            "content": f"Extract the question and headline from the text: {pair_str}, answer in json dict format, "
                       f"##Answer: <'q': , 'h': >"
        }
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])
    return decoded


start_t = time.time()


with open("data/formatted_answer_extracted.json", "r") as f:
    data = json.load(f)


formatted_data = {}


for i, (_id, pair) in enumerate(tqdm(data.items())):

    question = pair["question"]

    question = infer(question)

    print(question)

    if i == 10:
        break

print(formatted_data)

print(f"Takes {time.time() - start_t}s")
