from copy import deepcopy
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
from pathlib import Path
from typing import List, Dict
from functools import partial

sys.path.append("../../")
sys.path.append("../../../")
from tqdm import tqdm
import os
import torch
import sys
import pickle
import pandas as pd

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda


# from graphrag.query.structured_search.global_search.community_reports import (
#     GlobalCommunityReports,
# )
class GlobalCommunityReports:
    pass


random.seed(2025)

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-4B")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--divide", type=int, default=None)
args = parser.parse_args()
print(args)

# =============================================================================
# PART 1: DATA PREPARATION
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

backgroundContext = {
    "@vocab": str(SECURITY),
    "schema": str(SCHEMA),
    "xsd": str(XSD),
    "name": "schema:name",
    "identifier": "schema:identifier",
    "sector": "schema:sector",
    "industry": "schema:industry",
    "price": {"@id": "schema:price", "@type": "xsd:decimal"},
    "datePublished": {"@id": "schema:datePublished", "@type": "xsd:date"},
    "priceOf": {"@type": "@id"},
}

transactionContext = {
    "@vocab": str(SECURITY),
    "schema": str(SCHEMA),
    "transactions:": TRANSACTIONS_PREFIX,
    "users:": USER_PREFIX,
    # "prices:": PRICE_PREFIX,
    "summary:": SUMMARY_PREFIX,
    "isin:": ISIN_PREFIX,
    "xsd": str(XSD),
    "transactionValue": {"@type": "xsd:decimal"},
    # Define the datatype for your date predicate
    "transactionDate": {"@id": "security:transactionDate", "@type": "xsd:date"},
    "hasParticipant": {"@type": "@id"},
    "involvesSecurity": {"@type": "@id"},
}


# KG 1: Market Data
with open("./data/backgroundGraph.pkl", "rb") as file:
    backgroundKG = pickle.load(file)
# with open("./backgroundGraphExample.json", "w") as file:
#     file.write(
#         backgroundKG.serialize(format="json-ld", context=backgroundContext, indent=2)
#     )

# KG 2: Transaction History
with open("./data/clients.pkl", "rb") as file:
    clients = pickle.load(file)
# with open("./transactionGraphExample.json", "w") as file:
#     file.write(
#         clients[0][0].serialize(format="json-ld", context=transactionContext, indent=2)
#     )
# sys.exit()
transactionKGs = {
    next(transactionKG[: RDF.type : SECURITY.User])[len(USER_PREFIX) :]: transactionKG
    for client in clients
    for transactionKG in client
}
# print(list(transactionKGs.values()))


customerSubgraphCache = {}
backgroundSubgraphCache = {}


def getCustomerSubgraphUntilDate(user: str, endDate: str) -> Graph:
    if user is None:
        return Graph()
    global customerSubgraphCache, transactionKGs
    idStr = user + "_" + endDate
    if idStr in customerSubgraphCache:
        return customerSubgraphCache[idStr]
    graph = transactionKGs[user]
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

    for transactionURI in datapoints:
        priceDate = next(
            graph[transactionURI : SECURITY.transactionTimestamp :]
        ).toPython()

        if priceDate <= endDate:
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


def getBackgroundSubgraphUntilDate(endDate: str) -> Graph:
    global backgroundSubgraphCache, backgroundKG
    if endDate in backgroundSubgraphCache:
        return backgroundSubgraphCache[endDate]
    graph = backgroundKG
    subgraph = Graph()
    subgraph.bind("stock", SECURITY)
    subgraph.bind("schema", SCHEMA)

    endDate = datetime.strptime(endDate, "%Y-%m-%d").date()

    addedSecurities = set()

    datapoints = list(graph[: RDF.type : SECURITY.TenWeekPriceSummary])

    for priceObservation in datapoints:
        priceDate = next(graph[priceObservation : SECURITY.periodEndDate :]).toPython()

        if priceDate <= endDate:
            for s, p, o in graph.triples((priceObservation, None, None)):
                subgraph.add((s, p, o))

            securityURI = next(graph[priceObservation : SECURITY.priceOf :])
            if securityURI not in addedSecurities:
                for s, p, o in graph.triples((securityURI, None, None)):
                    subgraph.add((s, p, o))
                addedSecurities.add(securityURI)

    backgroundSubgraphCache[endDate] = subgraph
    return subgraph


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
retrievePipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=4096,
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=4096,
)
llm = HuggingFacePipeline(pipeline=pipe)
retrieve_llm = HuggingFacePipeline(pipeline=pipe)
print("Model loaded successfully.")

NODE_IDENTIFICATION_TEMPLATE = """
Your goal is to select a small, relevant subset of entities from a knowledge graph to help with a financial recommendation task.
Based on the user's request, identify the most relevant entities from the list provided.

Return only a bulleted list of the chosen entity URIs. Following this exact format:
- [ENTITY_URI_1]
- [ENTITY_URI_2]
- [ENTITY_URI_3]
- ...

## Entity List:
{entity_list}

## User Request:
"{request}"

## Relevant Entity URIs:
"""


def clean_qwen_output(text: str) -> str:
    """
    Removes the <think>...</think> block from the Qwen model's output.
    """
    print(
        "---------------------------- Identified -----------------------------" + text
    )
    # Use a regular expression to find and remove the think block
    # answerStartString = NODE_IDENTIFICATION_TEMPLATE[:-1][
    #     (NODE_IDENTIFICATION_TEMPLATE.rfind("\n") + 1) :
    # ]
    answerStartString = "## Relevant Entity URIs:"
    cleaned_text = text[(text.rfind(answerStartString) + len(answerStartString)) :]
    if cleaned_text.rfind("</think>") == -1:
        return ""
    cleaned_text = cleaned_text[(cleaned_text.rfind("</think>") + len("</think>")) :]
    # upToFirstItemText = cleaned_text[: text.find("http://")]
    # cleaned_text = text[
    #     (upToFirstItemText.rfind("\n")) : (
    #         len(upToFirstItemText) + cleaned_text[len(upToFirstItemText) :].find("\n")
    #     )
    # ]
    bullet_items = []
    foundBullet = False
    for line in cleaned_text.splitlines():
        if line.strip().startswith("-"):
            bullet_items.append(line.lstrip("- ").strip())
            foundBullet = True
        else:
            if foundBullet:
                break
    # bullet_items = [
    #     line.lstrip("- ").strip()
    #     for line in cleaned_text.splitlines()
    #     if line.strip().startswith("-")
    # ]

    if bullet_items:
        # If we found any bullet items, join them with a comma and space
        return ", ".join(bullet_items)
    # Also strip any leading/trailing whitespace that might be left over
    return None


MAX_ENTITIES = 50


class SubgraphRetriever(BaseRetriever):
    """
    Identifies key entities in a graph based on a request,
    constructs a subgraph around them, and returns it as JSON-LD.
    """

    graph: Graph
    llm: BaseLanguageModel
    candidate_query: str  # SPARQL query to find candidate nodes

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # 1. Get a list of all candidate entities using the provided SPARQL query
        candidate_uris = [str(row.s) for row in self.graph.query(self.candidate_query)]
        if not candidate_uris:
            return [
                Document(page_content="{}")
            ]  # Return empty JSON-LD if no candidates
        if len(candidate_uris) > MAX_ENTITIES * 4:
            candidate_uris = random.sample(candidate_uris, MAX_ENTITIES * 4)

        # Define the custom parser to clean the LLM output
        qwen_parser = RunnableLambda(clean_qwen_output)

        # 2. Use an LLM to identify the most relevant entities from the list
        # prompt = PromptTemplate.from_template(NODE_IDENTIFICATION_TEMPLATE)
        def applyTemp(inputs):
            return tokenizer.apply_chat_template(
                [
                    {
                        "content": NODE_IDENTIFICATION_TEMPLATE.format(**inputs),
                        "role": "user",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt = RunnableLambda(applyTemp)
        node_identification_chain = prompt | self.llm | qwen_parser | StrOutputParser()

        identified_nodes_str = node_identification_chain.invoke(
            {"entity_list": "\n".join(candidate_uris), "request": query}
        )
        print(
            "---------------------------- Identified -----------------------------\n"
            + identified_nodes_str
        )
        if identified_nodes_str is None:
            return None

        # 3. Programmatically build a CONSTRUCT query for the identified nodes
        node_list = [
            f"<{uri.strip()}>" for uri in identified_nodes_str.split(",") if uri.strip()
        ]
        if not node_list:
            return [Document(page_content="{}")]
        node_list = node_list[:MAX_ENTITIES]

        # This query constructs a subgraph including all triples where the identified
        # nodes are either the subject or the object.
        sparql_construct_query = f"""
        PREFIX ns: <http://securityTrade.com/ns#>
        CONSTRUCT {{ ?s ?p ?o }}
        WHERE {{
          VALUES ?node {{ {' '.join(node_list)} }}
          {{ ?node ?p ?o . BIND(?node as ?s) }}
          UNION
          {{ ?s ?p ?node . BIND(?node as ?o) }}
        }}
        """
        # print(sparql_construct_query)

        # 4. Execute the query to get the subgraph
        subgraph = self.graph.query(sparql_construct_query).graph
        if not subgraph:
            return [Document(page_content="{}")]

        # 5. Serialize the subgraph to JSON-LD and return as a single Document
        jsonld_output = subgraph.serialize(format="json-ld", indent=None)
        return [Document(page_content=jsonld_output)]


# =============================================================================
# PART 4: DYNAMIC RAG PIPELINE
# =============================================================================
SYSTEM_PROMPT_TASK = """You are an expert financial analyst AI. Your task is to analyze a user's transaction history and supplementary market data to provide personalized asset recommendations. The user will ask for recommendations for the next 6 months from a given "current date".

You MUST provide your response in the following format, and only this format:
[An introductory sentence]
- [ASSET_ISIN_1]
- [ASSET_ISIN_2]
- [ASSET_ISIN_3]"""
SYSTEM_PROMPT_BACKGROUND = """Here is the supplementary knowledge graph with asset information and historical prices in JSON-LD format:

```jsonld
{assets_jsonld}
```"""
SYSTEM_PROMPT_TRANSACTION = """Here is the user's transaction history in JSON-LD format:

```jsonld
{transactions_jsonld}
```"""

# --- Setup ---
# Assume 'llm', 'retrieval_llm', 'transactions_graph', 'assets_graph' are loaded

# 1. Define the SPARQL queries to find candidate nodes for each graph
TRANSACTIONS_CANDIDATE_QUERY = """
    PREFIX ns: <http://securityTrade.com/ns#>
    SELECT ?s WHERE {
      { ?s a ns:BuyTransaction } UNION { ?s a ns:SellTransaction }
    }
"""

ASSETS_CANDIDATE_QUERY = """
    PREFIX ns: <http://securityTrade.com/ns#>
    SELECT ?s WHERE {
      ?s a ns:TenWeekPriceSummary
    }
"""


# --- Define the Sequential Chain ---


def get_jsonld_from_docs(docs: List[Document]) -> str:
    return docs[0].page_content if docs else (None if docs is None else "{}")


def create_asset_query(input: dict) -> str:
    """Creates a new query for the asset retriever, combining the original question
    with the context from the retrieved transaction history."""
    return f"""
    Original user question: '{input["user_question"]}'
    
    Context from the user's transaction history is provided below. Identify the securities (ISINs) involved
    and find the most relevant price history for them to help answer the original question.
    
    Transaction History:
    {input["transactions_jsonld"]} 
    """


def format_qwen_chat_template(input_dict: dict, tokenizer) -> str:
    """
    Formats the final prompt for Qwen3 using the tokenizer's chat template.
    """
    # Build the list of messages in the required format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TASK},
        {
            "role": "system",
            "content": SYSTEM_PROMPT_BACKGROUND.format(
                assets_jsonld=input_dict["assets_jsonld"]
            ),
        },
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TRANSACTION.format(
                transactions_jsonld=input_dict["transactions_jsonld"]
            ),
        },
        # The user's request goes into the final 'user' role message
        {"role": "user", "content": input_dict["user_question"]},
    ]

    # Use the tokenizer to apply the chat template.
    # tokenize=False returns a single formatted string.
    # add_generation_prompt=True adds the required tokens to signal the model to start generating.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def ragCall(
    prompts,
    getTransactionData: bool = True,
    getMarketData: bool = True,
):
    global llm
    user_prompt = prompts[-1]["content"]
    currentDate = re.search(r"current date is ([^,]+),", user_prompt).group(1)
    userSearch = re.search(r"users::([^\"]+)\"", prompts[2]["content"])
    user = userSearch.group(1) if userSearch else None

    transactions_graph = getCustomerSubgraphUntilDate(user, currentDate)
    assets_graph = getBackgroundSubgraphUntilDate(currentDate)

    retriever_transactions = SubgraphRetriever(
        graph=transactions_graph,
        llm=retrieve_llm,
        candidate_query=TRANSACTIONS_CANDIDATE_QUERY,
    )
    retriever_assets = SubgraphRetriever(
        graph=assets_graph, llm=retrieve_llm, candidate_query=ASSETS_CANDIDATE_QUERY
    )

    qwen_formatter = RunnableLambda(
        partial(format_qwen_chat_template, tokenizer=tokenizer)
    )
    # Invoke the first step
    retrieved_transactions = (
        (
            (lambda inputs: user_prompt) | retriever_transactions | get_jsonld_from_docs
        ).invoke({"user_question": user_prompt})
        if getTransactionData
        else lambda _: "Empty"
    )

    if retrieved_transactions == None:
        print("--- Retrieval failed at transaction step. Halting. ---")
        return None

    retrieved_assets = (
        ((create_asset_query | retriever_assets | get_jsonld_from_docs)).invoke(
            {
                "user_question": user_prompt,
                "transactions_jsonld": getTransactionData,
            }
        )
        if getMarketData
        else lambda _: "Empty"
    )

    if retrieved_assets == None:
        print("--- Retrieval failed at asset step. Halting. ---")
        return None

    rag_chain = (
        # {
        #     "transactions_jsonld": (
        #         (
        #             (lambda inputs: inputs["user_question"])
        #             | retriever_transactions
        #             | get_jsonld_from_docs
        #         )
        #         if getTransactionData
        #         else lambda _: "Empty"
        #     ),
        #     "user_question": (lambda inputs: inputs["user_question"]),
        # }
        # | RunnablePassthrough.assign(
        #     assets_jsonld=(
        #         (create_asset_query | retriever_assets | get_jsonld_from_docs)
        #         if getMarketData
        #         else lambda _: "Empty"
        #     )
        # )
        # |
        qwen_formatter
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(
        {
            "user_question": user_prompt,
            "transactions_jsonld": retrieved_transactions,
            "assets_jsonld": retrieved_assets,
        }
    )

    return response


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

    truePositives = defaultdict(lambda: [])
    falsePositives = defaultdict(lambda: [])
    falseNegatives = defaultdict(lambda: [])
    hits = defaultdict(lambda: [[] for _ in range(10)])
    mrr = defaultdict(lambda: [])
    numDatapoints = 0
    for date, data in tqdm(dataset.items()):
        if saveResponses:
            responses[str(date)] = []
        numDatePoints = 0
        for index, dataPoint in enumerate(tqdm(data, leave=False)):
            if (
                saveResponses
                or not (str(date) in responses)
                or len(responses[str(date)]) <= index
            ):
                saveResponses = True
                response = ragCall(
                    dataPoint["prompt"],
                    not ("Transaction" in ignoreData),
                    not ("Background" in ignoreData),
                )

                responses[str(date)].append(response)

                with open(responsesPath, "w") as file:
                    json.dump(responses, file)
            else:
                response = responses[str(date)][index]
            print(f"---------------- RESPONSE --------------\n{response}")
            if response is None:
                continue
            endThinkString = "</think>"
            endThinkIndex = response.rfind(endThinkString)
            if endThinkIndex == -1:
                print("Output Did not complete thinking")
                continue
            fullResponse = response
            response = response[(endThinkIndex + len(endThinkString)) :]

            goals = dataPoint[goalName]
            print(f"---------------- GOALS --------------\n{goals}")

            recommendations = re.findall("(?=\n-([^\n]+))", response)
            formatFollowed = len(recommendations) > 0
            truePositives[date].append(0)
            falsePositives[date].append(0)
            falseNegatives[date].append(0)
            subResponse = response
            if formatFollowed:
                subResponse = "".join(recommendations)
                falsePositives[date][-1] += len(recommendations)
            rank = -1
            for goal in goals:
                if goal in subResponse:
                    truePositives[date][-1] += 1
                    print(f"{goal} found in response.")
                    if formatFollowed:
                        falsePositives[date][-1] -= 1
                        for recommendationIndex in range(
                            len(recommendations)
                            if rank < 0
                            else min(len(recommendations), rank)
                        ):
                            if goal in recommendations[recommendationIndex]:
                                rank = recommendationIndex
                    else:
                        rank = 0
                        falseNegatives[date][-1] += len(goals) - (goals.index(goal) + 1)
                        break
                else:
                    falseNegatives[date][-1] += 1
                    print(f"{goal} not found in response.")
            if len(tokenizer.encode(fullResponse, add_special_tokens=True)) > 4090 and (rank < 0 or rank > 2):
                truePositives[date].pop()
                falsePositives[date].pop()
                falseNegatives[date].pop()
                print("Output Did not complete after thinking")
                continue
            falseNegatives[date][-1] = max(
                0, min(20 - truePositives[date][-1], falseNegatives[date][-1])
            )
            mrr[date].append(0)
            for i in range(len(hits[date])):
                hits[date][i].append(0)
            if rank >= 0:
                mrr[date][-1] += 1 / (rank + 1)
                if len(hits[date]) < len(recommendations):
                    # hits[date] += [deepcopy(hits[date][-1])] * (
                    #     len(recommendations) - len(hits[date])
                    # )
                    hits[date].extend(
                        [
                            deepcopy(hits[date][-1])
                            for _ in range(len(recommendations) - len(hits[date]))
                        ]
                    )
                for i in range(rank, len(hits[date]), 1):
                    hits[date][i][-1] += 1
            print(f"truePositives[date][-1]: {truePositives[date][-1]}")
            print(f"falsePositives[date][-1]: {falsePositives[date][-1]}")
            print(f"falseNegatives[date][-1]: {falseNegatives[date][-1]}")
            print(f"Hits@: {[hities[-1] for hities in hits[date]]}")
            numDatePoints += 1
        if numDatePoints == 0:
            continue
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
                precision=sum(truePositives[date])
                / max(sum(truePositives[date]) + sum(falsePositives[date]), 1),
                recall=sum(truePositives[date])
                / max(sum(truePositives[date]) + sum(falseNegatives[date]), 1),
                mrr=sum(mrr[date]) / numDatePoints,
                hits="\n".join(
                    [
                        "Hits@{}: {}".format(
                            hitIndex + 1,
                            sum(hitCounts) / numDatePoints,
                        )
                        for hitIndex, hitCounts in enumerate(hits[date])
                    ]
                ),
            )
        )
        numDatapoints += numDatePoints

    if saveResponses:
        with open(responsesPath, "w") as file:
            json.dump(responses, file)

    def getSumOfDictVals(dictionary):
        return sum([sum(val) for val in dictionary.values()])

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
                "Hits@{}: {} - {}".format(
                    hitIndex + 1,
                    hitVal := sum(
                        [
                            (
                                sum(hitList[hitIndex])
                                if hitIndex < len(hitList)
                                else sum(hitList[-1])
                            )
                            for hitList in hits.values()
                        ]
                    )
                    / numDatapoints,
                    math.sqrt(hitVal * (1 - hitVal) / numDatapoints),
                )
                for hitIndex in range(max(len(hitList) for hitList in hits.values()))
            ]
        ),
    )


dataPath = "./data/testFinDatasetSmall.json"
with open(dataPath, "r") as file:
    testDataset = json.load(file)
for data in testDataset.values():
    random.shuffle(data)
if args.divide:
    testDataset = {
        date: data[: math.ceil(len(data) / args.divide)]
        for date, data in testDataset.items()
    }
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
