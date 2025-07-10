import datasets
import argparse
import json
import sys
import re
from collections import defaultdict

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


def runTests(dataset, goalName="completion", ignoreData="", name=None):
    if name is None:
        name = (
            args.base_model_path
            + "_"
            + args.lora_path.replace("/", "-")
            + "_"
            + args.data
            + "_ignore"
            + ignoreData
        )
    responses = {}
    responsesPath = "responses/" + name + ".json"
    saveResponses = True
    if os.path.exists(responsesPath):
        with open(responsesPath, "r") as file:
            responses = json.load(file)
        saveResponses = False

    truePositives = defaultdict(lambda: 0)
    falsePositives = defaultdict(lambda: 0)
    falseNegatives = defaultdict(lambda: 0)
    hits = defaultdict(lambda: [0] * 10)
    mrr = defaultdict(lambda: 0)
    numDatapoints = 0
    for date, data in tqdm(dataset.items()):
        if saveResponses:
            responses.append([])
        for index, dataPoint in enumerate(tqdm(data, leave=False)):
            if saveResponses:
                if "Background" in ignoreData:
                    dataPoint["prompt"][1][
                        "content"
                    ] = "Here is the supplementary knowledge graph with asset information and historical prices in JSON-LD format:\n\n```jsonld\nEmpty\n```"
                if "Transaction" in ignoreData:
                    dataPoint["prompt"][1][
                        "content"
                    ] = "Here is the user's transaction history in JSON-LD format:\n\n```jsonld\nEmpty\n```"
                if "gemma" in args.base_model_path:
                    dataPoint["prompt"] = [
                        {
                            "content": "\n".join(
                                [turn["content"] for turn in dataPoint["prompt"]]
                            ),
                            "role": "user",
                        }
                    ]

                text = tokenizer.apply_chat_template(
                    dataPoint["prompt"], tokenize=False, add_generation_prompt=True
                )

                print(f"---------------- PROMPT --------------\n{text}")

                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                generated_ids = model.generate(**model_inputs, max_new_tokens=4096)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]

                response = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                responses[date].append(response)
            else:
                response = responses[date][index]
            print(f"---------------- RESPONSE --------------\n{response}")

            goals = dataPoint[goalName]
            print(f"---------------- GOALS --------------\n{goals}")

            recommendations = re.findall("(?=\n-([^\n]+))", response)
            formatFollowed = len(recommendations) > 0
            subResponse = response
            if formatFollowed:
                subResponse = "".join(recommendations)
                falsePositives[date] += len(recommendations)
            rank = -1
            for goal in goals:
                if goal in subResponse:
                    truePositives[date] += 1
                    print(f"{goal} found in response.")
                    if formatFollowed:
                        falsePositives[date] -= 1
                        for recommendationIndex in range(
                            len(recommendations)
                            if rank < 0
                            else min(len(recommendations), rank)
                        ):
                            if goal in recommendations[recommendationIndex]:
                                rank = recommendationIndex
                    else:
                        rank = 0
                        falseNegatives[date] += len(goals) - (goals.index(goal) + 1)
                        break
                else:
                    falseNegatives[date] += 1
                    print(f"{goal} not found in response.")
            falseNegatives[date] = max(
                0, min(20 - truePositives[date], falseNegatives[date])
            )
            if rank >= 0:
                mrr[date] += 1 / (rank + 1)
                if len(hits[date]) < len(recommendations):
                    hits[date] += [hits[date][-1]] * (
                        len(recommendations) - len(hits[date])
                    )
                for i in range(rank, len(hits[date]), 1):
                    hits[date][i] += 1
            print(f"truePositives[date]: {truePositives[date]}")
            print(f"falsePositives[date]: {falsePositives[date]}")
            print(f"falseNegatives[date]: {falseNegatives[date]}")
            print(f"Hits@: {hits[date]}")
        numDatePoints = len(dataset)
        (
            "\nFor date: {date}\n"
            "Number of Tests: {num_tests}\n"
            "Precision: {precision}\n"
            "Recall: {recall}\n"
            "MRR: {mrr}\n"
            "{hits}"
        ).format(
            date=date,
            num_tests=numDatePoints,
            precision=truePositives[date]
            / (truePositives[date] + falsePositives[date]),
            recall=truePositives[date] / (truePositives[date] + falseNegatives[date]),
            mrr=mrr[date] / numDatePoints,
            hits="\n".join(
                [
                    "Hits@{}: {}".format(
                        hitIndex + 1,
                        hitCount / numDatePoints,
                    )
                    for hitIndex, hitCount in hits[date]
                ]
            ),
        )
        numDatapoints += numDatePoints

    with open(responsesPath, "w") as file:
        json.dump(responses, file)

    def getSumOfDictVals(dictionary):
        return sum(dictionary.values())

    return (
        "\nOverall Stats\n"
        "Number of Tests: {num_tests}\n"
        "Precision: {precision}\n"
        "Recall: {recall}\n"
        "MRR: {mrr}\n"
        "{hits}"
    ).format(
        num_tests=numDatapoints,
        precision=getSumOfDictVals(truePositives)
        / (getSumOfDictVals(truePositives) + getSumOfDictVals(falsePositives)),
        recall=getSumOfDictVals(truePositives)
        / (getSumOfDictVals(truePositives) + getSumOfDictVals(falseNegatives)),
        mrr=getSumOfDictVals(mrr) / numDatapoints,
        hits="\n".join(
            [
                "Hits@{}: {}".format(
                    hitIndex + 1,
                    sum(
                        [
                            (
                                hitList[hitIndex]
                                if hitIndex < len(hitList)
                                else hitList[-1]
                            )
                            for hitList in hits.values()
                        ]
                    )
                    / numDatapoints,
                )
                for hitIndex in range(max(len(hitList) for hitList in hits.values()))
            ]
        ),
    )


if args.data == "fin":
    with open("./data/testDataset.json", "r") as file:
        testDataset = json.load(file)
        print("Performing Hybrid Test:")
        print("Performing Overall Test:")
        print(f"Scores: {runTests(testDataset, "completion")}")
        print("Performing Profit Test:")
        print(f"Scores: {runTests(testDataset, "futurePurchases")}")
        print("Performing Adherence Test:")
        print(f"Scores: {runTests(testDataset, "profitableAssets")}")
        print("Performing no Background Test:")
        print("Performing Overall Test:")
        print(f"Scores: {runTests(testDataset, "completion", "Background")}")
        print("Performing Profit Test:")
        print(f"Scores: {runTests(testDataset, "futurePurchases", "Background")}")
        print("Performing Adherence Test:")
        print(f"Scores: {runTests(testDataset, "profitableAssets", "Background")}")
        print("Performing no Transaction Test:")
        print("Performing Overall Test:")
        print(f"Scores: {runTests(testDataset, "completion", "Transaction")}")
        print("Performing Profit Test:")
        print(f"Scores: {runTests(testDataset, "futurePurchases", "Transaction")}")
        print("Performing Adherence Test:")
        print(f"Scores: {runTests(testDataset, "profitableAssets", "Transaction")}")
        print("Performing no Data Test:")
        print("Performing Overall Test:")
        print(f"Scores: {runTests(testDataset, "completion", "BackgroundTransaction")}")
        print("Performing Profit Test:")
        print(
            f"Scores: {runTests(testDataset, "futurePurchases", "BackgroundTransaction")}"
        )
        print("Performing Adherence Test:")
        print(
            f"Scores: {runTests(testDataset, "profitableAssets", "BackgroundTransaction")}"
        )
