import pickle
import pandas as pd
import numpy as np
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD
from datetime import datetime
from tqdm import tqdm
import os

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


def createClients():

    CUSTOMER_BIAS_COLUMNS = ["customerType", "riskLevel", "investmentCapacity"]

    customerInformationDF = pd.read_csv(
        "./FAR-Trans/customer_information.csv",
        index_col="customerID",
        usecols=["customerID"] + CUSTOMER_BIAS_COLUMNS,
    )
    transactionsDF = pd.read_csv(
        "./FAR-Trans/transactions.csv", index_col="transactionID"
    )

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


def createBackgroundGraph():
    assetInformationDF = pd.read_csv("./FAR-Trans/asset_information.csv")
    assetInformationDF.drop_duplicates(subset="ISIN", inplace=True)
    closePricesDF = pd.read_csv("./FAR-Trans/close_prices.csv")

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


backgroundGraphPath = "./backgroundGraph.pkl"
if not os.path.exists(backgroundGraphPath):
    backgroundGraph = createBackgroundGraph()
    with open(backgroundGraphPath, "wb") as file:
        pickle.dump(backgroundGraph, file)
else:
    with open(backgroundGraphPath, "rb") as file:
        backgroundGraph = pickle.load(file)


def get_subgraph_until_date(graph: Graph, endDate: str) -> Graph:
    # 1. Setup the new subgraph
    subgraph = Graph()
    subgraph.bind("ex", EX)
    subgraph.bind("schema", SCHEMA)

    # Parse the end date for comparison
    end_date = datetime.strptime(endDate, "%Y-%m-%d").date()

    # A set to keep track of securities we've already added to avoid duplication
    added_securities = set()

    # 2. Find and filter all price observations
    # We query for the price observation, its date, and the security it belongs to.
    for price_obs, _, price_date_literal, security_uri in graph.triples_choices(
        (None, RDF.type, EX.PriceObservation),  # Subject is a PriceObservation
        (None, SCHEMA.datePublished, None),  # Get its date
        (None, EX.priceOf, None),  # Get the security it's for
    ):
        # Convert the literal date from the graph into a Python date object
        current_price_date = price_date_literal.toPython()

        # 3. If the date is within the desired range, copy the data
        if current_price_date <= end_date:
            # Copy all triples related to this specific price observation
            for s, p, o in graph.triples((price_obs, None, None)):
                subgraph.add((s, p, o))

            # If we haven't processed this security yet, copy its descriptive info
            if security_uri not in added_securities:
                for s, p, o in graph.triples((security_uri, None, None)):
                    subgraph.add((s, p, o))
                added_securities.add(security_uri)

    return subgraph
