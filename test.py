import datasets
import argparse
import json
import sys
import re

sys.path.append("../../")
sys.path.append("../../../")
from tqdm import tqdm
import os
import torch
import sys


from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--data", type=str, default="fin")
args = parser.parse_args()
print(args)

# ============= Extract model name from the path. The name is used for saving results. =============
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):  # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str  # mainly for base model
        exp_name = model_name


# ============= Generate responses =============
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path, torch_dtype=torch.float16
).to(device)
if args.lora_path is not None:
    model = PeftModel.from_pretrained(
        model, args.lora_path, torch_dtype=torch.float16
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)


def runTests(dataset):
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    hits = [0] * 10
    mrr = 0
    for dataPoint in tqdm(dataset):
        text = tokenizer.apply_chat_template(
            dataPoint["prompt"], tokenize=False, add_generation_prompt=True
        )

        print(f"---------------- PROMPT --------------\n{text}")

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=4096)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"---------------- RESPONSE --------------\n{response}")

        completion = dataPoint["completion"][0]["content"]
        print(f"---------------- COMPLETION --------------\n{completion}")

        recommendations = re.findall("(?=\n-([^\n]+))", response)
        formatFollowed = len(recommendations) > 0
        subResponse = response
        if formatFollowed:
            subResponse = "".join(recommendations)
            falsePositives += len(recommendations)
        rank = -1
        for goal in dataPoint["goal"]:
            if goal in subResponse:
                truePositives += 1
                print(f"{goal} found in response.")
                if formatFollowed:
                    falsePositives -= 1
                    for recommendationIndex in range(
                        len(recommendations)
                        if rank < 0
                        else min(len(recommendations), rank)
                    ):
                        if goal in recommendations[recommendationIndex]:
                            rank = recommendationIndex
                else:
                    rank = 0
                    falseNegatives += len(dataPoint["goal"]) - (
                        dataPoint["goal"].index(goal) + 1
                    )
                    break
            else:
                falseNegatives += 1
                print(f"{goal} not found in response.")
        if rank >= 0:
            mrr += 1 / (rank + 1)
            if len(hits) < len(recommendations):
                hits += [hits[-1]] * (len(recommendations) - len(hits))
            for i in range(rank, len(hits), 1):
                hits[i] += 1
        print(f"truePositives: {truePositives}")
        print(f"falsePositives: {falsePositives}")
        print(f"falseNegatives: {falseNegatives}")
        print(f"Hits@: {hits}")
    numDatapoints = len(dataset)
    return f"\n Number of Tests: {numDatapoints}\nPrecision: {truePositives/(truePositives+falsePositives)}\nRecall: {truePositives/(truePositives+falseNegatives)}\nMRR: {mrr/numDatapoints}\n{"\n".join([f"Hits@{index+1}: {hitCount/numDatapoints}" for index, hitCount in enumerate(hits)])})"


if args.data == "fin":
    with open("./data/testDataset.json", "r") as file:
        print("Performing Test:")
        print(f"Scores: {runTests(json.load(file))}")
