import datasets
import argparse
import random
import math
import json
import sys
import re
from collections import defaultdict
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD
from datetime import datetime
from itertools import chain
import yaml
import subprocess
from pathlib import Path

sys.path.append("../../")
sys.path.append("../../../")
from tqdm import tqdm
import os
import torch
import sys
import pickle
import pandas as pd

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.context_builder.conversation_history import ConversationHistory
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.language_model.manager import ModelManager
from graphrag.cli.index import index_cli
from graphrag.config.enums import IndexingMethod
from graphrag.language_model.protocol.base import ChatModel


# from graphrag.query.structured_search.global_search.community_reports import (
#     GlobalCommunityReports,
# )
class GlobalCommunityReports:
    pass


random.seed(2025)

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--smaller", action=argparse.BooleanOptionalAction)
args = parser.parse_args()
print(args)

# =============================================================================
# PART 1: DATA PREPARATION (RDFLib to CSV)
# =============================================================================

# Define your namespaces
SECURITY_PREFIX = "http://securityTrade.com/"
USER_PREFIX = SECURITY_PREFIX + "users/"
TRANSACTIONS_PREFIX = SECURITY_PREFIX + "transactions/"
PRICE_PREFIX = SECURITY_PREFIX + "prices/"
SUMMARY_PREFIX = SECURITY_PREFIX + "summary/"
ISIN_PREFIX = "urn:isin:"
SECURITY = Namespace(SECURITY_PREFIX + "ns#")
SCHEMA = Namespace("http://schema.org/")


def getCustomerSubgraphUntilDate(graph: Graph, endDate: str) -> Graph:
    subgraph = Graph()
    subgraph.bind("stock", SECURITY)
    subgraph.bind("schema", SCHEMA)

    endDate = datetime.strptime(endDate, "%Y-%m-%d").date()

    addedEntities = set()

    datapoints = list(
        chain(
            graph[: RDF.type : SECURITY.BuyTransaction],
            graph[: RDF.type : SECURITY.SellTransaction],
        )
    )

    for transactionURI in random.sample(
        datapoints,
        len(datapoints),
    ):
        priceDate = next(
            graph[transactionURI : SECURITY.transactionTimestamp :]
        ).toPython()

        if priceDate <= endDate.date():
            for s, p, o in graph.triples((transactionURI, None, None)):
                subgraph.add((s, p, o))

            def addEntity(relation):
                entity = next(graph[transactionURI:relation:])
                if entity not in addedEntities:
                    for s, p, o in graph.triples((entity, None, None)):
                        subgraph.add((s, p, o))
                    addedEntities.add(entity)

            addEntity(SECURITY.involvesSecurity)
            addEntity(SECURITY.hasParticipant)

    return subgraph


def getBackgroundSubgraphUntilDate(graph: Graph, endDate: str) -> Graph:
    subgraph = Graph()
    subgraph.bind("stock", SECURITY)
    subgraph.bind("schema", SCHEMA)

    endDate = datetime.strptime(endDate, "%Y-%m-%d").date()

    addedSecurities = set()

    datapoints = list(graph[: RDF.type : SECURITY.TenWeekPriceSummary])

    for priceObservation in random.sample(datapoints, len(datapoints)):
        priceDate = next(graph[priceObservation : SECURITY.periodEndDate :]).toPython()

        if priceDate <= endDate.date():
            for s, p, o in graph.triples((priceObservation, None, None)):
                subgraph.add((s, p, o))

            securityURI = next(graph[priceObservation : SECURITY.priceOf :])
            if securityURI not in addedSecurities:
                for s, p, o in graph.triples((securityURI, None, None)):
                    subgraph.add((s, p, o))
                addedSecurities.add(securityURI)

    return subgraph


# KG 1: Market Data
with open("./data/backgroundGraph.pkl", "rb") as file:
    backgroundKG = pickle.load(file)

# KG 2: Transaction History
with open("./data/clients.pkl", "rb") as file:
    clients = pickle.load(file)
transactionKGs = {
    next(transactionKG[: RDF.type : SECURITY.User]): transactionKG
    for client in clients
    for transactionKG in client
}

BACKGROUND_KG_INPUT_DIR = "./data/GraphRAG/backgroundGraphs/"
TRANSACTION_KGs_INPUT_DIR = "./data/GraphRAG/transactionGraphs/"


def getBackgroundDir(dateString):
    path = BACKGROUND_KG_INPUT_DIR + "/" + dateString
    os.makedirs(path, exist_ok=True)
    return path


def getTransactionDir(dateString, user):
    path = TRANSACTION_KGs_INPUT_DIR + "/" + dateString + "/" + user
    os.makedirs(path, exist_ok=True)
    return path


def prepareCSVDataForDate(dateString: str, user: str):
    global backgroundKG, transactionKGs

    def convertKGtoGraphRAGFormat(dir_path, kg):
        os.makedirs(dir_path, exist_ok=True)
        entities = {}
        relationships = []

        for s, p, o in kg:
            source_id, target_id = str(s), str(o)
            s_splitter = "/" if "/" in s else ":"
            o_splitter = "/" if "/" in o else ":"
            p_splitter = "/" if "/" in p else ":"
            if source_id not in entities:
                entities[source_id] = {
                    "id": source_id,
                    "title": s.split(s_splitter)[-1],
                }
            if isinstance(o, URIRef) and target_id not in entities:
                entities[target_id] = {
                    "id": target_id,
                    "title": o.split(o_splitter)[-1],
                }
            elif isinstance(o, Literal):
                entities[source_id][p.split(p_splitter)[-1]] = str(o)
            if isinstance(o, URIRef):
                relationships.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "relationship": p.split(p_splitter)[-1],
                    }
                )

        pd.DataFrame(list(entities.values())).to_csv(
            f"{dir_path}/entities.csv", index=False
        )
        pd.DataFrame(relationships).to_csv(f"{dir_path}/relationships.csv", index=False)

    backgroundPath = getBackgroundDir(dateString)
    if not os.path.exists(backgroundPath):
        convertKGtoGraphRAGFormat(
            backgroundPath, getBackgroundSubgraphUntilDate(backgroundKG, dateString)
        )
    if user is not None:
        transactionPath = getTransactionDir(dateString, user)
        if not os.path.exists(transactionPath):
            convertKGtoGraphRAGFormat(
                transactionPath,
                getBackgroundSubgraphUntilDate(transactionKGs[user], dateString),
            )
    else:
        transactionPath = None
    return backgroundPath, transactionPath


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


# ============= Load Model =============
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path,
    torch_dtype=torch.float16,
    rope_scaling={"type": "yarn", "factor": 4.0},
).to(device)
if args.lora_path is not None:
    model = PeftModel.from_pretrained(
        model, args.lora_path, torch_dtype=torch.float16
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
print("Model loaded successfully.")


class LLMOutput:
    """A custom wrapper to use a local Hugging Face model with GraphRAG."""

    def __init__(self, response, prompt_tokens, completion_tokens):
        self.response = response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class LLMInput:
    """A custom wrapper to use a local Hugging Face model with GraphRAG."""

    def __init__(self, messages):
        self.messages = messages


# --- Create the Custom Wrapper Class ---
class HuggingFaceWrapper(ChatModel):
    """A custom wrapper to use a local Hugging Face model with GraphRAG."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, input: LLMInput, clean: bool = True) -> LLMOutput:
        text = self.tokenizer.apply_chat_template(
            input.messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        # Clean the <think> blocks from the final generated text
        if clean:
            response = re.sub(
                r"<think>.*?</think>", "", response, flags=re.DOTALL
            ).strip()

        return LLMOutput(
            response=response,
            prompt_tokens=num_tokens(text),
            completion_tokens=num_tokens(response),
        )


# =============================================================================
# PART 3: GRAPHRAG SETUP & INDEXING
# =============================================================================
rootGraphRAGDir = "./graphRAG"
os.makedirs(rootGraphRAGDir, exist_ok=True)


def setup_and_run_indexing(dateString, user):
    """Creates config files and runs the GraphRAG indexing process."""
    print("--- Part 3: Setting up and running GraphRAG indexing ---")
    backgroundInputPath, transactionInputPath = prepareCSVDataForDate(dateString, user)
    backgroundOutputPath = backgroundInputPath.replace("./data", rootGraphRAGDir)
    transactionOutputPath = transactionInputPath.replace("./data", rootGraphRAGDir)
    configs = {
        "transaction_kg_{}_{}.yml".format(dateString, user): {
            "input_dir": transactionInputPath,
            "output_dir": transactionOutputPath,
        },
        "background_kg_{}.yml".format(dateString): {
            "input_dir": backgroundInputPath,
            "output_dir": backgroundOutputPath,
        },
    }
    for filename, paths in configs.items():
        fileDir = "./conf/GraphRAG/"
        filePath = fileDir + filename
        config_data = {
            "llm": {
                "type": "static_response",
                "response": "This is a mock response for indexing.",
            },
            "storage": {"type": "file", "base_dir": paths["output_dir"]},
            "input": {
                "type": "file",
                "base_dir": paths["input_dir"],
                "entity": {
                    "file_type": "csv",
                    "source": "entities.csv",
                    "id": "id",
                    "title": "title",
                },
                "relationship": {
                    "file_type": "csv",
                    "source": "relationships.csv",
                    "source_id": "source",
                    "target_id": "target",
                },
            },
        }
        if not os.path.exists(filePath):
            os.makedirs(fileDir, exist_ok=True)
            os.makedirs(paths["output_dir"], exist_ok=True)
            with open(filePath, "w") as f:
                yaml.dump(config_data, f)

            print(f"Running indexing for {filePath}...")
            index_cli(
                Path(rootGraphRAGDir),
                IndexingMethod.Standard,
                False,
                False,
                False,
                Path(filePath),
                False,
                True,
                Path(paths["output_dir"]),
            )
            # subprocess.run(
            #     [
            #         "graphrag",
            #         "index",
            #         "--root",
            #         ".",
            #         "-c",
            #         filePath,
            #     ],
            #     check=True,
            # )
    print("Indexing complete for all graphs.\n")
    return backgroundOutputPath, transactionOutputPath


# =============================================================================
# PART 4: DYNAMIC RAG PIPELINE
# =============================================================================
SYSTEM_PROMPT_TASK = """You are an expert financial analyst AI. Your task is to analyze a user's transaction history and supplementary market data to provide personalized asset recommendations. The user will ask for recommendations for the next 6 months from a given "current date".

You MUST provide your response in the following format, and only this format:
[An introductory sentence]
- [ASSET_ISIN_1]
- [ASSET_ISIN_2]
- [ASSET_ISIN_3]"""
SYSTEM_PROMPT_BACKGROUND = """Here is the supplementary data with asset information and historical prices:

```
{}
```"""
SYSTEM_PROMPT_TRANSACTION = """Here is the user's transaction history:

```
{}
```"""


def setup_query_engine(storage_dir: str, llm_wrapper):
    """Loads an indexed graph and sets up its query engine."""
    engine = LocalSearch(
        llm=llm_wrapper,
        context_builder=GlobalCommunityReports(
            llm=llm_wrapper,
            community_reports=None,
            conversation_history=ConversationHistory(max_turns=5),
        ),
        community_level=2,
        response_type="multiple paragraphs",
    )
    engine.load_context(storage_dir)
    return engine


def generate_search_query(user_prompt: str, llm_wrapper) -> str:
    """Uses the LLM to distill a user prompt into a concise search query."""
    messages = [
        {
            "role": "system",
            "content": "You are an expert at creating search queries. Given a user's request, extract the core intent into a short query for a financial knowledge graph.",
        },
        {"role": "user", "content": user_prompt},
    ]
    llm_input = LLMInput(messages=messages)
    response = llm_wrapper(llm_input)
    return response.response


llm_wrapper = HuggingFaceWrapper(model, tokenizer)


def run_financial_query(
    prompts: list,
    getTransactionData: bool = True,
    getMarketData: bool = True,
):
    global llm_wrapper
    user_prompt = prompts[-1]["content"]
    currentDate = re.search(r"current date is ([^,]+),", user_prompt).group(1)
    userSearch = re.search(r"users::([^\"]+)\"", prompts[2]["content"])
    user = userSearch.group(1) if userSearch else None
    backgroundOutputPath, transactionOutputPath = setup_and_run_indexing(
        currentDate, user
    )

    """Runs the full two-step RAG pipeline."""
    print("--- Part 4: Executing Dynamic RAG Pipeline ---")

    # 1. Generate the initial search query from the user's prompt
    initial_query = generate_search_query(user_prompt, llm_wrapper)
    print(f"Dynamically generated search query: '{initial_query}'")

    if user is not None and getTransactionData:
        # 2. First Retrieval (Transaction History) - STEP 1
        transaction_engine = setup_query_engine(transactionOutputPath, llm_wrapper)
        print("\n-> Step 1: Retrieving from Transaction History KG...")
        response_transactions = transaction_engine.run(initial_query)
        context_transactions = response_transactions.response
    else:
        context_transactions = "Empty"

    if getMarketData:
        # 3. Second Retrieval (Market Data) - STEP 2
        market_engine = setup_query_engine(backgroundOutputPath, llm_wrapper)
        print("-> Step 2: Retrieving from Market Data KG...")
        # Augment the query with context from the user's transaction history
        augmented_query = f"{initial_query}\n\nRelevant transaction history context: {context_transactions}"
        response_market = market_engine.run(augmented_query)
        context_market = response_market.response
    else:
        context_market = "Empty"

    # 4. Final Prompt Assembly and Generation
    print("-> Step 3: Assembling final prompt and generating answer...")

    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TASK},
        {"role": "system", "content": SYSTEM_PROMPT_BACKGROUND.format(context_market)},
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TRANSACTION.format(context_transactions),
        },
        {"role": "user", "content": user_prompt},
    ]

    final_input = LLMInput(messages=final_messages)
    final_response = llm_wrapper(final_input, clean=False)
    return final_response.response


# =============================================================================
# PART 5: MAIN EXECUTION
# =============================================================================


def runTests(dataset, goalName="completion", ignoreData="", name=None):
    if name is None:
        name = (
            "RAG_"
            + args.base_model_path
            + (
                ("_" + args.lora_path.replace("/", "-"))
                if args.lora_path is not None
                else ""
            )
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
    else:
        responsesFolder = responsesPath[: responsesPath.rfind("/")]
        if not os.path.exists(responsesFolder):
            os.makedirs(responsesFolder)

    truePositives = defaultdict(lambda: 0)
    falsePositives = defaultdict(lambda: 0)
    falseNegatives = defaultdict(lambda: 0)
    hits = defaultdict(lambda: [0] * 10)
    mrr = defaultdict(lambda: 0)
    numDatapoints = 0
    for date, data in tqdm(dataset.items()):
        if saveResponses:
            responses[str(date)] = []
        numDatePoints = 0
        for index, dataPoint in enumerate(tqdm(data, leave=False)):
            if saveResponses:
                response = run_financial_query(
                    dataPoint["prompt"],
                    "Transaction" in ignoreData,
                    "Background" in ignoreData,
                )

                responses[str(date)].append(response)
            else:
                response = responses[str(date)][index]
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
            numDatePoints += 1
        print(
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
                / max(truePositives[date] + falsePositives[date], 1),
                recall=truePositives[date]
                / max(truePositives[date] + falseNegatives[date], 1),
                mrr=mrr[date] / numDatePoints,
                hits="\n".join(
                    [
                        "Hits@{}: {}".format(
                            hitIndex + 1,
                            hitCount / numDatePoints,
                        )
                        for hitIndex, hitCount in enumerate(hits[date])
                    ]
                ),
            )
        )
        numDatapoints += numDatePoints

    if saveResponses:
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


dataPath = "./data/testFinDatasetSmall.json"
with open(dataPath, "r") as file:
    testDataset = json.load(file)
    if args.smaller:
        testDataset = {
            date: random.sample(data, math.ceil(len(data) / 10))
            for date, data in testDataset.items()
        }
    testDataset = {date: random.sample(data, 1) for date, data in testDataset.items()}
    print("Performing Hybrid Test:")
    print("Performing Overall Test:")
    print("Scores: {}".format(runTests(testDataset, "completion")))
    print("Performing Adherence Test:")
    print("Scores: {}".format(runTests(testDataset, "futurePurchases")))
    print("Performing Profit Test:")
    print("Scores: {}".format(runTests(testDataset, "profitableAssets")))
    # print("Performing no Background Test:")
    # print("Performing Overall Test:")
    # print("Scores: {}".format(runTests(testDataset, "completion", "Background")))
    # print("Performing Adherence Test:")
    # print("Scores: {}".format(runTests(testDataset, "futurePurchases", "Background")))
    # print("Performing Profit Test:")
    # print("Scores: {}".format(runTests(testDataset, "profitableAssets", "Background")))
    # print("Performing no Transaction Test:")
    # print("Performing Overall Test:")
    # print("Scores: {}".format(runTests(testDataset, "completion", "Transaction")))
    # print("Performing Adherence Test:")
    # print("Scores: {}".format(runTests(testDataset, "futurePurchases", "Transaction")))
    # print("Performing Profit Test:")
    # print("Scores: {}".format(runTests(testDataset, "profitableAssets", "Transaction")))
