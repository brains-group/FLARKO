import pickle
import pandas as pd
import numpy as np
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD
from datetime import datetime, timedelta
from tqdm import tqdm
from itertools import chain
from dateutil.relativedelta import relativedelta
import os
import json

np.random.seed(2025)

# Define your namespaces
SECURITY_PREFIX = "http://securityTrade.com/"
USER_PREFIX = SECURITY_PREFIX + "users/"
ISIN_PREFIX = "urn:isin:"
SECURITY = Namespace(SECURITY_PREFIX + "ns#")
SCHEMA = Namespace("http://schema.org/")

# ------------------------- Transaction Graphs Creation --------------------------

transactionContext = {
    "@vocab": str(SECURITY),
    "schema": str(SCHEMA),
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

            transactionURI = URIRef(
                SECURITY_PREFIX + "transactions/" + str(transactionID)
            )
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

            if pd.notna(row.get("assetName")):
                backgroundGraph.add(
                    (securityURI, SCHEMA.name, Literal(row["assetName"]))
                )
            if pd.notna(row.get("assetCategory")):
                backgroundGraph.add(
                    (securityURI, SECURITY.assetCategory, Literal(row["assetCategory"]))
                )
            if pd.notna(row.get("assetSubCategory")):
                backgroundGraph.add(
                    (
                        securityURI,
                        SECURITY.assetSubCategory,
                        Literal(row["assetSubCategory"]),
                    )
                )
            if pd.notna(row.get("sector")):
                backgroundGraph.add(
                    (securityURI, SCHEMA.sector, Literal(row["sector"]))
                )
            if pd.notna(row.get("industry")):
                backgroundGraph.add(
                    (securityURI, SCHEMA.industry, Literal(row["industry"]))
                )

    for index, row in tqdm(closePricesDF.iterrows()):
        isin = row["ISIN"]
        timestamp = row["timestamp"]
        closePrice = row["closePrice"]

        if pd.notna(isin) and pd.notna(timestamp) and pd.notna(closePrice):
            securityURI = URIRef(f"urn:isin:{isin}")

            # Create a unique URI for each price observation
            priceURI = URIRef(f"http://example.com/prices/{isin}_{timestamp}")

            # Add the price observation data
            backgroundGraph.add((priceURI, RDF.type, SECURITY.PriceObservation))
            backgroundGraph.add((priceURI, SECURITY.priceOf, securityURI))
            backgroundGraph.add(
                (priceURI, SCHEMA.price, Literal(closePrice, datatype=XSD.decimal))
            )
            backgroundGraph.add(
                (priceURI, SCHEMA.datePublished, Literal(timestamp, datatype=XSD.date))
            )

    # print(
    #     len(
    #         backgroundGraph.serialize(
    #             format="json-ld", context=backgroundContext, indent=2
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


def getBackgroundSubgraphUntilDate(graph: Graph, endDate: datetime) -> Graph:
    subgraph = Graph()
    subgraph.bind("stock", SECURITY)
    subgraph.bind("schema", SCHEMA)

    # endDate = datetime.strptime(endDate, "%Y-%m-%d").date()

    addedSecurities = set()

    for priceObservation in tqdm(graph[: RDF.type : SECURITY.PriceObservation]):
        priceDate = next(graph[priceObservation : SCHEMA.datePublished :]).toPython()

        if priceDate <= endDate:
            for s, p, o in graph.triples((priceObservation, None, None)):
                subgraph.add((s, p, o))

            securityURI = next(graph[priceObservation : SECURITY.priceOf :])
            if securityURI not in addedSecurities:
                for s, p, o in graph.triples((securityURI, None, None)):
                    subgraph.add((s, p, o))
                addedSecurities.add(securityURI)

    return subgraph


# print("------------------------------")
# print(
#     len(
#         getBackgroundSubgraphUntilDate(backgroundGraph, "2018-06-01").serialize(
#             format="json-ld", context=backgroundContext, indent=2
#         )
#     )
# )


def getCustomerSubgraphUntilDate(graph: Graph, endDate: datetime) -> Graph:
    subgraph = Graph()
    subgraph.bind("stock", SECURITY)
    subgraph.bind("schema", SCHEMA)

    # endDate = datetime.strptime(endDate, "%Y-%m-%d").date()

    addedEntities = set()

    for transactionURI in tqdm(
        chain(
            graph[: RDF.type : SECURITY.BuyTransaction],
            graph[: RDF.type : SECURITY.SellTransaction],
        )
    ):
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


# for client in clients:
#     for graph in client:
#         graphLen = len(graph)
#         if graphLen > 100:
#             print(graphLen)
# print("------------------------------")
# print(
#     len(
#         getCustomerSubgraphUntilDate(clients[4][10], "2018-06-01").serialize(
#             format="json-ld", context=transactionContext, indent=2
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

customerSubgraphStrings = {}
backgroundSubgraphStrings = {}


def getCustomerSubgraphUntilDateString(graph: Graph, endDate: datetime):
    global customerSubgraphStrings
    if endDate not in customerSubgraphStrings:
        customerSubgraphStrings[endDate] = getCustomerSubgraphUntilDate(
            graph, endDate
        ).serialize(format="json-ld", context=transactionContext, indent=2)
    return customerSubgraphStrings[endDate]


def getBackgroundSubgraphUntilDateString(graph: Graph, endDate: datetime):
    global backgroundSubgraphStrings
    if endDate not in backgroundSubgraphStrings:
        backgroundSubgraphStrings[endDate] = getBackgroundSubgraphUntilDate(
            graph, endDate
        ).serialize(format="json-ld", context=backgroundContext, indent=2)
    return backgroundSubgraphStrings[endDate]


def generate_kto_data(
    transactionGraph: str,
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
    futureDate = currDate + relativedelta(months=+6)
    customerID = next(transactionGraph[: RDF.type : SECURITY.User])[len(USER_PREFIX) :]

    # 2. Find assets the user ACTUALLY bought in the next 6 months
    futureTransactions = transactionsDF[
        (transactionsDF["customerID"] == customerID)
        & (pd.to_datetime(transactionsDF["timestamp"]) > currDate)
        & (pd.to_datetime(transactionsDF["timestamp"]) <= futureDate)
        & (transactionsDF["transactionType"] == "Buy")
    ]
    future_purchases = set(futureTransactions["ISIN"].unique())
    if len(future_purchases) == 0:
        return []

    # 3. Find assets that were PROFITABLE in the next 6 months
    goodAssets = set()
    for isin in future_purchases:
        try:
            start_price_row = (
                closePricesDF[
                    (closePricesDF["ISIN"] == isin)
                    & (pd.to_datetime(closePricesDF["timestamp"]) <= currDate)
                ]
                .sort_values(by="timestamp", ascending=False)
                .iloc[0]
            )

            end_price_row = (
                closePricesDF[
                    (closePricesDF["ISIN"] == isin)
                    & (pd.to_datetime(closePricesDF["timestamp"]) > currDate)
                    & (pd.to_datetime(closePricesDF["timestamp"]) <= futureDate)
                ]
                .sort_values(by="timestamp", ascending=False)
                .iloc[0]
            )

            if end_price_row["closePrice"] > start_price_row["closePrice"]:
                goodAssets.add(isin)
        except IndexError:
            # Not enough price data to determine profitability
            continue
    if len(goodAssets) == 0:
        return []

    # 4. Determine GOOD and BAD assets based on your criteria
    badAssets = set(closePricesDF["ISIN"].unique()) - goodAssets

    # 5. Yield KTO data points
    # (Here you would generate the filtered KGs for the prompt)
    # prompt_background_kg = get_subgraph_until_date(...)
    # prompt_user_kg = get_transactions_subgraph_until_date(...)

    prompt = [
        {"content": SYSTEM_PROMPT_TASK, "role": "system"},
        {
            "content": SYSTEM_PROMPT_BACKGROUND.format(
                getBackgroundSubgraphUntilDateString(transactionGraph, currDate)
            ),
            "role": "system",
        },
        {
            "content": SYSTEM_PROMPT_TRANSACTION.format(
                getCustomerSubgraphUntilDateString(backgroundGraph, currDate)
            ),
            "role": "system",
        },
        {"content": USER_PROMPT.format(currDate.date()), "role": "user"},
    ]

    def createDatapoint(assets, label):
        return {
            "prompt": prompt,
            "completion": f"Here are my asset recommendations:\n- {"\n- ".join(assets)}",
            "label": label,
        }

    return [createDatapoint(goodAssets, True), createDatapoint(badAssets, False)]


ktoDataset = [[] for _ in range(len(clients))]

trainDateLimit = pd.to_datetime("2021-12-1") - relativedelta(months=+6)
startTrainDate = pd.to_datetime("2019-08-1")
for clientIndex, client in enumerate(tqdm(clients)):
    for currDate in tqdm(
        pd.date_range(trainDateLimit, startTrainDate, freq=timedelta(weeks=-4))
    ):
        for graph in tqdm(client):
            ktoDataset[clientIndex].extend(
                generate_kto_data(
                    graph,
                    currDate,
                )
            )

with open("finDataset.json", "w") as file:
    json.dump(ktoDataset, file, indent=4)

nonFederatedDataset = []
for client in ktoDataset.values():
    nonFederatedDataset.extend(client)

with open("nonFedFinDataset.json", "w") as file:
    json.dump(nonFederatedDataset, file, indent=4)
