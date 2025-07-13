import pickle
import random
import pandas as pd
import numpy as np
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD
from datetime import datetime, timedelta
from tqdm import tqdm
from itertools import chain
import os
import json

from transformers import AutoTokenizer

np.random.seed(2025)
random.seed(2025)

# Define your namespaces
SECURITY_PREFIX = "http://securityTrade.com/"
USER_PREFIX = SECURITY_PREFIX + "users/"
TRANSACTIONS_PREFIX = SECURITY_PREFIX + "transactions/"
PRICE_PREFIX = SECURITY_PREFIX + "prices/"
SUMMARY_PREFIX = SECURITY_PREFIX + "summary/"
ISIN_PREFIX = "urn:isin:"
SECURITY = Namespace(SECURITY_PREFIX + "ns#")
SCHEMA = Namespace("http://schema.org/")

# ------------------------- Transaction Graphs Creation --------------------------

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

transactionsDF = pd.read_csv("./FAR-Trans/transactions.csv", index_col="transactionID")


def createClients():

    CUSTOMER_BIAS_COLUMNS = ["customerType", "riskLevel", "investmentCapacity"]

    customerInformationDF = pd.read_csv(
        "./FAR-Trans/customer_information.csv",
        index_col="customerID",
        usecols=["customerID"] + CUSTOMER_BIAS_COLUMNS,
    )
    global transactionsDF

    print(transactionsDF.head())

    cutomerTransactionsDFGrouped = transactionsDF.groupby("customerID")
    print(cutomerTransactionsDFGrouped.get_group("00017496858921195E5A").head())
    print(len(cutomerTransactionsDFGrouped))
    print(customerInformationDF.shape)
    print(len(transactionsDF["customerID"].unique()))

    customerInformationDF = customerInformationDF.loc[
        ~customerInformationDF.index.duplicated(keep="first")
    ]
    customerInformationDF = customerInformationDF.map(
        lambda x: x.replace("Predicted_", "")
    )
    print(customerInformationDF.shape)
    print(customerInformationDF.head())

    uniqueCustomerTraits = {
        columnName: customerInformationDF[columnName].unique()
        for columnName in CUSTOMER_BIAS_COLUMNS
    }

    clients = [[] for _ in range(20)]
    biases = np.random.random_sample(
        (
            len(CUSTOMER_BIAS_COLUMNS),
            max(len(unique) for unique in uniqueCustomerTraits.values()),
            len(clients),
        )
    )
    biases = np.apply_along_axis(lambda x: x / np.sum(x), axis=2, arr=biases)
    print(biases[:, :, 0])
    print(np.sum(biases[2, 0, :]))

    def handleUser(g: Graph, customerID: str):
        user = URIRef(USER_PREFIX + customerID)
        userTriple = (user, RDF.type, SECURITY.User)
        if userTriple not in g:
            g.add(userTriple)
        return user

    def handleSecurity(g: Graph, isin: str):
        security = URIRef(ISIN_PREFIX + isin)
        securityTriple = (security, RDF.type, SECURITY.Security)
        if securityTriple not in g:
            g.add(securityTriple)
        return security

    def createTransactionGraph(transactions: pd.DataFrame):
        g = Graph()

        g.bind("stock", SECURITY)
        g.bind("schema", SCHEMA)

        for transactionID, transaction in transactions.iterrows():
            userURI = handleUser(g, transaction["customerID"])
            securityURI = handleSecurity(g, transaction["ISIN"])

            transactionURI = URIRef(TRANSACTIONS_PREFIX + str(transactionID))
            g.add(
                (
                    transactionURI,
                    RDF.type,
                    (
                        SECURITY.BuyTransaction
                        if transaction["transactionType"] == "Buy"
                        else SECURITY.SellTransaction
                    ),
                )
            )
            g.add((transactionURI, SECURITY.hasParticipant, userURI))
            g.add((transactionURI, SECURITY.involvesSecurity, securityURI))
            g.add(
                (
                    transactionURI,
                    SECURITY.transactionValue,
                    Literal(transaction["totalValue"], datatype=XSD.decimal),
                )
            )
            g.add(
                (
                    transactionURI,
                    SECURITY.transactionTimestamp,
                    Literal(
                        transaction["timestamp"],
                        datatype=XSD.date,
                    ),
                )
            )
        return g

    def getBiasIndexFromBiasValue(customerInformation: pd.DataFrame, columnName: str):
        return np.where(
            uniqueCustomerTraits[columnName] == customerInformation[columnName]
        )[0][0]

    for customerID, transactions in tqdm(cutomerTransactionsDFGrouped):
        customerInformation = customerInformationDF.loc[customerID]
        biasIndexes = [
            getBiasIndexFromBiasValue(customerInformation, columnName)
            for columnName in CUSTOMER_BIAS_COLUMNS
        ]
        # print(biasIndexes)
        # print(
        #     np.array(
        #         [
        #             biases[columnIndex, biasIndex, :]
        #             for columnIndex, biasIndex in zip(range(len(biasIndexes)), biasIndexes)
        #         ]
        #     )
        # )
        specBiases = np.sum(
            np.array(
                [
                    biases[columnIndex, biasIndex, :]
                    for columnIndex, biasIndex in zip(
                        range(len(biasIndexes)), biasIndexes
                    )
                ]
            ),
            axis=0,
        )
        specBiases = specBiases / np.sum(specBiases)
        # print(specBiases)
        # print(np.sum(specBiases))
        clientIndex = np.random.choice(np.arange(len(clients)), p=specBiases)
        # print(clientIndex)
        clients[clientIndex].append(createTransactionGraph(transactions))

    print([len(client) for client in clients])
    for clientIndex, client in enumerate(clients):
        clientStats = [[0 for _ in types] for types in uniqueCustomerTraits.values()]
        for graph in client:
            customerID = next(graph[: RDF.type : SECURITY.User])[len(USER_PREFIX) :]
            customerInformation = customerInformationDF.loc[customerID]
            for columnIndex, columnName in enumerate(CUSTOMER_BIAS_COLUMNS):
                clientStats[columnIndex][
                    getBiasIndexFromBiasValue(customerInformation, columnName)
                ] += 1
        print(str(clientIndex) + ":\n" + str(clientStats))

    return clients


clientsPath = "./clients.pkl"
if not os.path.exists(clientsPath):
    clients = createClients()
    with open(clientsPath, "wb") as file:
        pickle.dump(clients, file)
else:
    with open(clientsPath, "rb") as file:
        clients = pickle.load(file)


# -------------------- Background Graph Creation --------------------------------

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

closePricesDF = pd.read_csv("./FAR-Trans/close_prices.csv")
closePricesDF["timestamp"] = pd.to_datetime(closePricesDF["timestamp"])


def createBackgroundGraph():
    assetInformationDF = pd.read_csv("./FAR-Trans/asset_information.csv")
    assetInformationDF.drop_duplicates(subset="ISIN", inplace=True)
    global closePricesDF

    backgroundGraph = Graph()

    backgroundGraph.bind("stock", SECURITY)
    backgroundGraph.bind("schema", SCHEMA)

    for index, row in tqdm(assetInformationDF.iterrows()):
        isin = row["ISIN"]
        if pd.notna(isin):
            securityURI = URIRef(ISIN_PREFIX + isin)

            backgroundGraph.add((securityURI, RDF.type, SECURITY.Security))
            backgroundGraph.add((securityURI, SCHEMA.identifier, Literal(isin)))

            # if pd.notna(row.get("assetName")):
            #     backgroundGraph.add(
            #         (securityURI, SCHEMA.name, Literal(row["assetName"]))
            #     )
            if pd.notna(row.get("assetCategory")):
                backgroundGraph.add(
                    (securityURI, SECURITY.assetCategory, Literal(row["assetCategory"]))
                )
            # if pd.notna(row.get("assetSubCategory")):
            #     backgroundGraph.add(
            #         (
            #             securityURI,
            #             SECURITY.assetSubCategory,
            #             Literal(row["assetSubCategory"]),
            #         )
            #     )
            if pd.notna(row.get("sector")):
                backgroundGraph.add(
                    (securityURI, SCHEMA.sector, Literal(row["sector"]))
                )
            if pd.notna(row.get("industry")):
                backgroundGraph.add(
                    (securityURI, SCHEMA.industry, Literal(row["industry"]))
                )

    for isin, group in tqdm(closePricesDF.groupby("ISIN")):
        # Set the date as the index for resampling
        group = group.set_index("timestamp").sort_index()

        # Resample the data into 10-week periods ('10W')
        # .agg() lets us calculate multiple statistics at once
        summaries = (
            group["closePrice"]
            .resample("10W")
            .agg(
                {
                    "end_price": "last",  # The last price in the period
                    "avg_price": "mean",  # The average price
                    "high_price": "max",  # The highest price
                    "low_price": "min",  # The lowest price
                }
            )
        )

        # Drop any periods that have no data
        summaries.dropna(inplace=True)

        securityURI = URIRef(f"{ISIN_PREFIX}{isin}")

        # Add each summary period to the knowledge graph
        for end_date, summary in tqdm(summaries.iterrows(), leave=False):
            period_date_str = end_date.strftime("%Y-%m-%d")
            summaryURI = URIRef(f"{SUMMARY_PREFIX}{isin}_{period_date_str}")

            backgroundGraph.add((summaryURI, RDF.type, SECURITY.TenWeekPriceSummary))
            backgroundGraph.add((summaryURI, SECURITY.priceOf, securityURI))
            backgroundGraph.add(
                (
                    summaryURI,
                    SECURITY.periodEndDate,
                    Literal(period_date_str, datatype=XSD.date),
                )
            )

            backgroundGraph.add(
                (
                    summaryURI,
                    SECURITY.periodEndPrice,
                    Literal(summary["end_price"], datatype=XSD.decimal),
                )
            )
            backgroundGraph.add(
                (
                    summaryURI,
                    SECURITY.periodAveragePrice,
                    Literal(summary["avg_price"], datatype=XSD.decimal),
                )
            )
            backgroundGraph.add(
                (
                    summaryURI,
                    SECURITY.periodHighPrice,
                    Literal(summary["high_price"], datatype=XSD.decimal),
                )
            )
            backgroundGraph.add(
                (
                    summaryURI,
                    SECURITY.periodLowPrice,
                    Literal(summary["low_price"], datatype=XSD.decimal),
                )
            )

    # for index, row in tqdm(closePricesDF.iterrows()):
    #     isin = row["ISIN"]
    #     timestamp = row["timestamp"]
    #     closePrice = row["closePrice"]

    #     if pd.notna(isin) and pd.notna(timestamp) and pd.notna(closePrice):
    #         securityURI = URIRef(f"{ISIN_PREFIX}{isin}")

    #         # Create a unique URI for each price observation
    #         priceURI = URIRef(f"{PRICE_PREFIX}{isin}_{timestamp}")

    #         # Add the price observation data
    #         backgroundGraph.add((priceURI, RDF.type, SECURITY.PriceObservation))
    #         backgroundGraph.add((priceURI, SECURITY.priceOf, securityURI))
    #         backgroundGraph.add(
    #             (priceURI, SCHEMA.price, Literal(closePrice, datatype=XSD.decimal))
    #         )
    #         backgroundGraph.add(
    #             (priceURI, SCHEMA.datePublished, Literal(timestamp, datatype=XSD.date))
    #         )

    # print(
    #     len(
    #         backgroundGraph.serialize(
    #             format="json-ld", context=backgroundContext, indent=None
    #         )
    #     )
    # )

    return backgroundGraph


backgroundGraphPath = "./backgroundGraph.pkl"
if not os.path.exists(backgroundGraphPath):
    backgroundGraph = createBackgroundGraph()
    with open(backgroundGraphPath, "wb") as file:
        pickle.dump(backgroundGraph, file)
else:
    with open(backgroundGraphPath, "rb") as file:
        backgroundGraph = pickle.load(file)


# ----------------------------- LLM Input Creation ----------------------------------
MAX_TOTAL_GRAPH_LENGTH = 5000


def getBackgroundSubgraphUntilDate(
    graph: Graph, endDate: datetime, usedUpGraphLength: int = 0
) -> Graph:
    subgraph = Graph()
    subgraph.bind("stock", SECURITY)
    subgraph.bind("schema", SCHEMA)

    # endDate = datetime.strptime(endDate, "%Y-%m-%d").date()

    addedSecurities = set()

    datapoints = list(graph[: RDF.type : SECURITY.TenWeekPriceSummary])

    maxLength = MAX_TOTAL_GRAPH_LENGTH - usedUpGraphLength
    lengthUsed = 0
    for priceObservation in random.sample(datapoints, len(datapoints)):
        priceDate = next(graph[priceObservation : SECURITY.periodEndDate :]).toPython()

        if priceDate <= endDate.date():
            for s, p, o in graph.triples((priceObservation, None, None)):
                subgraph.add((s, p, o))
                lengthUsed += 1

            securityURI = next(graph[priceObservation : SECURITY.priceOf :])
            if securityURI not in addedSecurities:
                for s, p, o in graph.triples((securityURI, None, None)):
                    subgraph.add((s, p, o))
                    lengthUsed += 1
                addedSecurities.add(securityURI)

            if maxLength < lengthUsed:
                break

    return subgraph


# print("------------------------------")
# print(
#     len(
#         getBackgroundSubgraphUntilDate(backgroundGraph, "2018-06-01").serialize(
#             format="json-ld", context=backgroundContext, indent=None
#         )
#     )
# )


def getCustomerSubgraphUntilDate(
    graph: Graph, endDate: datetime, usedUpGraphLength: int = 0
) -> Graph:
    subgraph = Graph()
    subgraph.bind("stock", SECURITY)
    subgraph.bind("schema", SCHEMA)

    # endDate = datetime.strptime(endDate, "%Y-%m-%d").date()

    addedEntities = set()

    datapoints = list(
        chain(
            graph[: RDF.type : SECURITY.BuyTransaction],
            graph[: RDF.type : SECURITY.SellTransaction],
        )
    )

    maxLength = MAX_TOTAL_GRAPH_LENGTH - usedUpGraphLength
    lengthUsed = 0
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
                lengthUsed += 1

            def addEntity(relation):
                nonlocal lengthUsed
                entity = next(graph[transactionURI:relation:])
                if entity not in addedEntities:
                    for s, p, o in graph.triples((entity, None, None)):
                        subgraph.add((s, p, o))
                        lengthUsed += 1
                    addedEntities.add(entity)

            addEntity(SECURITY.involvesSecurity)
            addEntity(SECURITY.hasParticipant)
            if maxLength <= lengthUsed:
                break

    return subgraph


# for client in clients:
#     for graph in client:
#         graphLen = len(graph)
#         if graphLen > 100:
#             print(graphLen)
# print("------------------------------")
# print(
#     len(
#         getCustomerSubgraphUntilDate(clients[4][10], "2018-06-01").serialize(
#             format="json-ld", context=transactionContext, indent=None
#         )
#     )
# )

SYSTEM_PROMPT_TASK = """You are an expert financial analyst AI. Your task is to analyze a user's transaction history and supplementary market data to provide personalized asset recommendations. The user will ask for recommendations for the next 6 months from a given "current date".

You MUST provide your response in the following format, and only this format:
[An introductory sentence]
- [ASSET_ISIN_1]
- [ASSET_ISIN_2]
- [ASSET_ISIN_3]"""
SYSTEM_PROMPT_BACKGROUND = """Here is the supplementary knowledge graph with asset information and historical prices in JSON-LD format:

```jsonld
{}
```"""
SYSTEM_PROMPT_TRANSACTION = """Here is the user's transaction history in JSON-LD format:

```jsonld
{}
```"""
USER_PROMPT = """Considering all the provided data, and assuming the current date is {}, please provide a list of asset recommendations for my portfolio for the next 6 months."""

# customerSubgraphStrings = {}
# backgroundSubgraphStrings = {}

allAssets = set(closePricesDF["ISIN"].unique())
closePricesDF.set_index(["ISIN", "timestamp"], inplace=True)
closePricesDF.sort_index(inplace=True)

transactionsDF["timestamp"] = pd.to_datetime(transactionsDF["timestamp"])
transactionsDF.reset_index(inplace=True)
transactionsDF.set_index(["customerID", "timestamp"], inplace=True)
transactionsDF.sort_index(inplace=True)


# def getCustomerSubgraphUntilDateString(
#     graph: Graph, endDate: datetime, usedUpGraphLength: int = 0
# ):
#     # global customerSubgraphStrings
#     # if endDate not in customerSubgraphStrings:
#     #     customerSubgraphStrings[endDate] = getCustomerSubgraphUntilDate(
#     #         graph, endDate
#     #     ).serialize(format="json-ld", context=transactionContext, indent=None)
#     # return customerSubgraphStrings[endDate]
#     return getCustomerSubgraphUntilDate(graph, endDate, usedUpGraphLength).serialize(
#         format="json-ld", context=transactionContext, indent=None
#     )


# def getBackgroundSubgraphUntilDateString(
#     graph: Graph, endDate: datetime, usedUpGraphLength: int = 0
# ):
#     # global backgroundSubgraphStrings
#     # if endDate not in backgroundSubgraphStrings:
#     #     backgroundSubgraphStrings[endDate] = getBackgroundSubgraphUntilDate(
#     #         graph, endDate
#     #     ).serialize(format="json-ld", context=backgroundContext, indent=None)
#     # return backgroundSubgraphStrings[endDate]
#     return getBackgroundSubgraphUntilDate(graph, endDate, usedUpGraphLength).serialize(
#         format="json-ld", context=backgroundContext, indent=None
#     )


def getSubgraphsUntilDateStrings(
    transactionGraph: Graph, backgroundGraph: Graph, endDate: datetime
):
    transactionSubGraph = getCustomerSubgraphUntilDate(
        transactionGraph, endDate, MAX_TOTAL_GRAPH_LENGTH * 0.8
    )
    backgroundSubGraph = getBackgroundSubgraphUntilDate(
        backgroundGraph, endDate, len(transactionSubGraph)
    )
    # print("len(transactionSubGraph): " + str(len(transactionSubGraph)))
    # print("len(backgroundSubGraph): " + str(len(backgroundSubGraph)))
    return transactionSubGraph.serialize(
        format="json-ld", context=transactionContext, indent=None
    ), backgroundSubGraph.serialize(
        format="json-ld", context=backgroundContext, indent=None
    )


def generate_kto_data(
    transactionGraph: Graph,
    currDate: datetime,
    closePricesDF: pd.DataFrame = closePricesDF,
    transactionsDF: pd.DataFrame = transactionsDF,
):
    """
    Generates KTO training examples for a given user and evaluation date.

    Args:
        transactionGraph: The transaction graph of the user to generate data for.
        currDate: The 'current date' for the recommendation in 'YYYY-MM-DD' format.
        closePricesDF: DataFrame of ALL historical prices.
        transactionsDF: DataFrame of ALL user transactions.

    Yields:
        A dictionary for each KTO example containing the prompt and the completion.
    """
    # 1. Setup dates
    futureDate = currDate + timedelta(days=180)
    customerID = next(transactionGraph[: RDF.type : SECURITY.User])[len(USER_PREFIX) :]

    # 2. Find assets the user ACTUALLY bought in the next 6 months
    futureTransactions = transactionsDF.query(
        "customerID == @customerID and \
        timestamp > @currDate and \
        timestamp <= @futureDate and \
        transactionType == 'Buy'"
    )
    futurePurchases = set(futureTransactions["ISIN"].unique())
    if len(futurePurchases) == 0:
        return []

    # 3. Find assets that were PROFITABLE in the next 6 months
    goodAssets = set()
    for isin in futurePurchases:
        try:
            startPrice = closePricesDF.query(
                "ISIN == @isin and timestamp <= @currDate"
            ).iloc[-1]["closePrice"]

            endPrice = closePricesDF.query(
                "ISIN == @isin and timestamp > @currDate and timestamp <= @futureDate"
            ).iloc[-1]["closePrice"]

            if endPrice > startPrice:
                goodAssets.add(isin)
        except IndexError:
            # Not enough price data to determine profitability
            continue
    if len(goodAssets) == 0:
        return []

    # 4. Determine GOOD and BAD assets based on your criteria
    badAssets = allAssets - goodAssets

    maxRecommendations = 20
    if len(goodAssets) > maxRecommendations:
        goodAssets = random.sample(list(goodAssets), maxRecommendations)
    if len(badAssets) > maxRecommendations:
        badAssets = random.sample(list(badAssets), maxRecommendations)

    # 5. Yield KTO data points
    # (Here you would generate the filtered KGs for the prompt)
    # prompt_background_kg = get_subgraph_until_date(...)
    # prompt_user_kg = get_transactions_subgraph_until_date(...)

    transactionSubGraph, backgroundSubGraph = getSubgraphsUntilDateStrings(
        transactionGraph, backgroundGraph, currDate
    )
    prompt = [
        {"content": SYSTEM_PROMPT_TASK, "role": "system"},
        {
            "content": SYSTEM_PROMPT_BACKGROUND.format(backgroundSubGraph),
            "role": "system",
        },
        {
            "content": SYSTEM_PROMPT_TRANSACTION.format(transactionSubGraph),
            "role": "system",
        },
        {"content": USER_PROMPT.format(currDate.date()), "role": "user"},
    ]

    def createDatapoint(assets, label):
        return {
            "prompt": prompt,
            "completion": [
                {
                    "content": f"Here are my asset recommendations:\n- {"\n- ".join(assets)}",
                    "role": "assistant",
                }
            ],
            "label": label,
        }

    return [createDatapoint(goodAssets, True), createDatapoint(badAssets, False)]


def createTestDataset():
    ktoDataset = [[] for _ in range(len(clients))]

    trainDateLimit = pd.to_datetime("2021-12-1") - timedelta(days=180)
    startTrainDate = pd.to_datetime("2019-08-1")
    for clientIndex, client in enumerate(tqdm(clients)):
        for currDate in tqdm(
            pd.date_range(trainDateLimit, startTrainDate, freq=timedelta(weeks=-4)),
            leave=False,
        ):
            for graph in tqdm(random.sample(client, int(len(client) / 5)), leave=False):
                ktoDataset[clientIndex].extend(
                    generate_kto_data(
                        graph,
                        currDate,
                    )
                )
    return ktoDataset


# tokenizer = AutoTokenizer.from_pretrained(
#     "Qwen/Qwen3-0.6B",
#     use_fast=False,
#     # padding_side=cfg.train.padding_side,
# )
# test = clients[0][0]
# for client in clients:
#     for graph in client:
#         if len(graph) > len(test):
#             test = graph
# transactionSubGraph, backgroundSubGraph = getSubgraphsUntilDateStrings(
#     test, backgroundGraph, pd.to_datetime("2021-12-1")
# )
# text = tokenizer.apply_chat_template(
#     [
#         {"content": SYSTEM_PROMPT_TASK, "role": "system"},
#         {
#             "content": SYSTEM_PROMPT_BACKGROUND.format(backgroundSubGraph),
#             "role": "system",
#         },
#         {
#             "content": SYSTEM_PROMPT_TRANSACTION.format(transactionSubGraph),
#             "role": "system",
#         },
#         {
#             "content": USER_PROMPT.format(pd.to_datetime("2021-12-1").date()),
#             "role": "user",
#         },
#     ],
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
# )
# model_inputs = tokenizer([text], return_tensors="pt")
# print('len(model_inputs["input_ids"]): ' + str(len(model_inputs["input_ids"][0])))
# exit()

ktoDatasetPath = "./finDataset.json"
if not os.path.exists(ktoDatasetPath):
    ktoDataset = createTestDataset()
    with open(ktoDatasetPath, "w") as file:
        json.dump(ktoDataset, file, indent=4)
else:
    with open(ktoDatasetPath, "r") as file:
        ktoDataset = json.load(file)

ktoNonfedDatasetPath = "./nonFedFinDataset.json"
if not os.path.exists(ktoNonfedDatasetPath):
    nonFederatedDataset = []
    for client in ktoDataset:
        nonFederatedDataset.extend(client)
    with open(ktoNonfedDatasetPath, "w") as file:
        json.dump(nonFederatedDataset, file, indent=4)
else:
    with open(ktoNonfedDatasetPath, "r") as file:
        nonFederatedDataset = json.load(file)

print(len(nonFederatedDataset))


# -------------------------------- Create Test Dataset ------------------------------

closePricesDF.reset_index(inplace=True)
closePricesDF["timestamp"] = pd.to_datetime(closePricesDF["timestamp"])
closePricesDF.sort_values(["ISIN", "timestamp"], inplace=True)


def generate_test_data(
    transactionGraph: Graph,
    currDate: datetime,
    closePricesDF: pd.DataFrame = closePricesDF,
    transactionsDF: pd.DataFrame = transactionsDF,
):
    """
    Generates KTO training examples for a given user and evaluation date.

    Args:
        transactionGraph: The transaction graph of the user to generate data for.
        currDate: The 'current date' for the recommendation in 'YYYY-MM-DD' format.
        closePricesDF: DataFrame of ALL historical prices.
        transactionsDF: DataFrame of ALL user transactions.

    Yields:
        A dictionary for each KTO example containing the prompt and the completion.
    """
    # 1. Setup dates
    futureDate = currDate + timedelta(days=180)
    customerID = next(transactionGraph[: RDF.type : SECURITY.User])[len(USER_PREFIX) :]

    # 2. Find assets the user ACTUALLY bought in the next 6 months
    futureTransactions = transactionsDF.query(
        "customerID == @customerID and \
        timestamp > @currDate and \
        timestamp <= @futureDate and \
        transactionType == 'Buy'"
    )
    futurePurchases = set(futureTransactions["ISIN"].unique())
    if len(futurePurchases) == 0:
        return []

    # print("Data type of 'timestamp' column:", closePricesDF.dtypes)
    # print("Data type of comparison date 'currDate':", type(currDate))

    # 4. Get Start Prices using a boolean mask on the sorted columns
    startPricesDF = closePricesDF[closePricesDF["timestamp"] <= currDate]
    startPrices = (
        startPricesDF.groupby("ISIN").last()["closePrice"].rename("startPrice")
    )

    # 5. Get End Prices using a boolean mask
    futurePricesDF = closePricesDF[
        (closePricesDF["timestamp"] > currDate)
        & (closePricesDF["timestamp"] <= futureDate)
    ]
    endPrices = futurePricesDF.groupby("ISIN").last()["closePrice"].rename("endPrice")

    # 5. Get the final set of ISINs
    profitDF = pd.concat([startPrices, endPrices], axis=1).dropna()
    profitable_assets_series = profitDF[profitDF["endPrice"] > profitDF["startPrice"]]
    profitableAssets = set(profitable_assets_series.index)
    if len(profitableAssets) == 0:
        return []

    goodAssets = futurePurchases.intersection(profitableAssets)
    if len(goodAssets) == 0:
        return []

    transactionSubGraph, backgroundSubGraph = getSubgraphsUntilDateStrings(
        transactionGraph, backgroundGraph, currDate
    )
    prompt = [
        {"content": SYSTEM_PROMPT_TASK, "role": "system"},
        {
            "content": SYSTEM_PROMPT_BACKGROUND.format(backgroundSubGraph),
            "role": "system",
        },
        {
            "content": SYSTEM_PROMPT_TRANSACTION.format(transactionSubGraph),
            "role": "system",
        },
        {"content": USER_PROMPT.format(currDate.date()), "role": "user"},
    ]

    return [
        {
            "prompt": prompt,
            "futurePurchases": list(futurePurchases),
            "profitableAssets": list(profitableAssets),
            "completion": list(goodAssets),
        }
    ]


def createTestDataset():
    testDateLimit = pd.to_datetime("2022-11-29") - timedelta(days=180)
    startTestDate = pd.to_datetime("2021-12-1")

    dates = pd.date_range(testDateLimit, startTestDate, freq=timedelta(weeks=-2))
    testDataset = {str(date): [] for date in dates}

    for client in tqdm(clients):
        for currDate in tqdm(
            dates,
            leave=False,
        ):
            for graph in tqdm(client, leave=False):
                testDataset[str(currDate)].extend(
                    generate_test_data(
                        graph,
                        currDate,
                    )
                )
    return testDataset


testDatasetPath = "./testFinDataset.json"
if not os.path.exists(testDatasetPath):
    picklePath = testDatasetPath[:-5] + ".pkl"
    if os.path.exists(picklePath):
        with open(picklePath, "rb") as file:
            testDataset = pickle.load(file)

        # # Fixing mistake with dataset
        # def fixDatapoint(datapoint):
        #     datapoint["futurePurchases"] = list(datapoint["futurePurchases"])
        #     datapoint["profitableAssets"] = list(datapoint["profitableAssets"])
        #     datapoint["completion"] = list(datapoint["completion"])
        #     return datapoint

        # testDataset = {
        #     date: [
        #         fixDatapoint(datapoint)
        #         for datapoint in dateData
        #         if (not (datapoint == [] or len(datapoint["completion"]) == 0))
        #     ]
        #     for date, dateData in testDataset.items()
        # }
    else:
        testDataset = createTestDataset()
        with open(picklePath, "wb") as file:
            pickle.dump(testDataset, file)
    with open(testDatasetPath, "w") as file:
        json.dump(testDataset, file, indent=4)
else:
    with open(testDatasetPath, "r") as file:
        testDataset = json.load(file)

print(len(testDataset))
for dateData in testDataset.values():
    print(len(dateData))

smallTestDatasetPath = "./testFinDatasetSmall.json"
if not os.path.exists(smallTestDatasetPath):
    smallTestDataset = {
        date: random.sample(dateData, int(len(dateData) * 0.01))
        for date, dateData in testDataset.items()
    }
    with open(smallTestDatasetPath, "w") as file:
        json.dump(smallTestDataset, file, indent=4)
else:
    with open(smallTestDatasetPath, "r") as file:
        smallTestDataset = json.load(file)

print(len(smallTestDataset))
for dateData in smallTestDataset.values():
    print(len(dateData))
