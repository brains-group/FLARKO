import pandas as pd
import numpy as np
from rdflib import Graph, Literal, RDF, URIRef, Namespace

# Define your namespaces
STOCK = Namespace("http://stockTrade.com/ns#")
SCHEMA = Namespace("http://schema.org/")

np.random.seed(2025)

CUSTOMER_BIAS_COLUMNS = ["customerType", "riskLevel", "investmentCapacity"]

assetInformationDF = pd.read_csv("./FAR-Trans/asset_information.csv", index_col="ISIN")
closePricesDF = pd.read_csv(
    "./FAR-Trans/close_prices.csv", index_col=["ISIN", "timestamp"]
)
customerInformationDF = pd.read_csv(
    "./FAR-Trans/customer_information.csv",
    index_col="customerID",
    usecols=["customerID"] + CUSTOMER_BIAS_COLUMNS,
)
limitPricesDF = pd.read_csv("./FAR-Trans/limit_prices.csv", index_col="ISIN")
marketsDF = pd.read_csv("./FAR-Trans/markets.csv", index_col="marketID")
transactionsDF = pd.read_csv("./FAR-Trans/transactions.csv", index_col="transactionID")

print(transactionsDF.head())

cutomerTransactionsDFGrouped = transactionsDF.groupby("customerID")
print(cutomerTransactionsDFGrouped.get_group("00017496858921195E5A").head())
print(len(cutomerTransactionsDFGrouped))
print(customerInformationDF.shape)
print(len(transactionsDF["customerID"].unique()))

customerInformationDF = customerInformationDF.loc[
    ~customerInformationDF.index.duplicated(keep="first")
]
customerInformationDF = customerInformationDF.map(lambda x: x.replace("Predicted_", ""))
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
        max(len(unique) for unique in uniqueCustomerTraits),
        len(clients),
    )
)
biases = np.apply_along_axis(lambda x: x / np.sum(x), axis=2, arr=biases)
print(biases[:, :, 0])
print(np.sum(biases[2, 0, :]))

def handleUser():
    pass

def createTransactionGraph(transactions):
    g = Graph()

    g.bind("stock", STOCK)
    g.bind("schema", SCHEMA)

    for transactionID, transaction in transactions.iterrows():

        print("-------------")
        print(transaction)
        print(transactionID)
        print(transaction["customerID"])
        break


for customerID, transactions in cutomerTransactionsDFGrouped:
    customerInformation = customerInformationDF.loc[customerID]
    biasIndexes = [
        np.where(uniqueCustomerTraits[columnName] == customerInformation[columnName])[
            0
        ][0]
        for columnName in CUSTOMER_BIAS_COLUMNS
    ]
    print(biasIndexes)
    print(
        np.array(
            [
                biases[columnIndex, biasIndex, :]
                for columnIndex, biasIndex in zip(range(len(biasIndexes)), biasIndexes)
            ]
        )
    )
    specBiases = np.sum(
        np.array(
            [
                biases[columnIndex, biasIndex, :]
                for columnIndex, biasIndex in zip(range(len(biasIndexes)), biasIndexes)
            ]
        ),
        axis=0,
    )
    specBiases = specBiases / np.sum(specBiases)
    print(specBiases)
    print(np.sum(specBiases))
    clientIndex = np.random.choice(np.arange(len(clients)), p=specBiases)
    print(clientIndex)
    clients[clientIndex].append(createTransactionGraph(transactions))
    break
