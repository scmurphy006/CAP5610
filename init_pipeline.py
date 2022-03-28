import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score

import conversation

REQUESTED_SCORE = "PII Requested Score"

PATTERN_MATCHES = "PII Pattern Matches"

# My gross regex patterns that could probably be written much more cleanly
patterns = {
            "ADDRESS_PATTERN": re.compile(r"(unit|suite|apt|street|st|road|rd|drive|dr|lane|ln|avenue|ave|boulevard|blvd|highway|hwy|township|twp|north|south|east|west|apt.)"),
            "STATES_PATTERN": re.compile(r"(,?) (AL|AK|AZ|AR|CA|CZ|CO|CT|DE|DC|FL|GA|GU|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|PR|RI|SC|SD|TN|TX|UT|VT|VI|VA|WA|WV|WI|WY)"),
            "US_PHONE_PATTERN": re.compile(r"(?i)((\+?1(\.|-|\s)?)?)\s*((\(?\d{3}\)?(\.|-|\s*)?)?)\s*(\d{3}(\.|-|\s*)?)\s*(\d{4}\s*(((x|ext)\.?(ension)?)\s*\d*)?)"),
            "EMAIL_PATTERN": re.compile(r"([\w\.-]+)@([\da-zA-Z\.-]+)\.([a-zA-Z\.]{2,6})"),
            "PLATE_PATTERN": re.compile(r"([A-Z]{1,3}-[A-Z]{1,2}-[0-9]{1,4})"),
            "NAME_PATTERN": re.compile(r"[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z]([a-z]+|\.)"),
            "SSN_PATTERN": re.compile(r"(?!000|666)[0-8]\d{2}(-|\s)(?!00)\d{2}(-|\s)(?!0000)\d{4}"),
            "CC_PATTERN": re.compile(r"(\d{15,16}\s)"),
            "VISA_PATTERN": re.compile(r"4[0-9]{12}(?:[0-9]{3})?"),
            "MC_PATTERN": re.compile(r"(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}"),
            "AMEX_PATTERN": re.compile(r"3[47][0-9]{13}"),
            "DISCOVER_PATTERN": re.compile(r"6(?:011|5[0-9]{2})[0-9]{12}")}

# Not used in the current Pipeline
# Used to measure the similarity between a conversation and someone requesting PII. If we get around to it
question_words = ["address", "name", "birth", "phone", "country", "county", "district", "subcountry", "parish", "village", "community", "gps", "lat", "lon", "coord", "location", "house", "compount", "school", "social", "network", "email", "age", "religion", "occupation", "work", "years", "old"]

# Show plots for DBSCAN and epsilon determination
INCLUDE_PLOTS = True

# True = Attempt to cluster based on PII detection and Requesting PII score. BETA
# False = Simple, regex based PII detection
ENHANCED = True

def process_train_data(test_data, num_records=80000):
    """Process the training dataset and just see how accurate the regex patterns are at determining PII

    Args:
        test_data (dictionary): Empty dictionary
        num_records (int): The number of records to use for this test. If not using enhanced then just leave this at default

    Returns:
        dictionary: The processed data
    """
    data_labels = []
    data = pd.read_csv('train_10k.csv', header=0).to_dict()
    index = 1 
    emails = [{"index": i, "text": data["Text"][i], "label": data["Labels"][i]} for i in data["Text"]]
    convos_len = num_records
    print("Processessing test: (0/" + str(int(convos_len)) + ")", end="")
    for i in range(num_records):
        data_labels.append(emails[i])
        test_data[REQUESTED_SCORE].append(conversation.Conversation(emails[i]["text"]).do_search(question_words))
        pii_len = 0
        for j in patterns:
            if re.search(patterns[j], emails[i]["text"].lower()) is not None:
                pii_len += 1

        value = pii_len
        if pii_len > 1:
            value = math.log2(pii_len)
        test_data[PATTERN_MATCHES].append(value)
        index += 1
        print(
                    "\rProcessessing test: "
                    + "("
                    + str(index)
                    + "/"
                    + str(int(convos_len))
                    + ")",
                    end="",
                    flush=True,
                )
    return data_labels


def process_test_data(train_data):
    """Process the testing dataset and just see how accurate the regex patterns are at determining PII

    Args:
        train_data (dictionary): The blank dictionary to fill
    """
    with open("Z:\STIR\ML\IGDDPIIShare\MySQL Exports\data_10.json", encoding="utf-8") as fh:
        content = json.load(fh, strict=False)
        convos = {}

        for i in content:
            if i["ConversationID"] not in convos:
                convos[i["ConversationID"]] = []
            convos[i["ConversationID"]].append(i["Message"])

        index = 1
        convos_len = len(convos)
        print("Processessing train: (0/" + str(int(convos_len)) + ")", end="")
        for i in convos:
            c = "\n".join(convos[i])
            train_data[REQUESTED_SCORE].append(conversation.Conversation(c).do_search(question_words))
            pii_len = 0
            for j in patterns:
                pii_len += len(patterns[j].findall(c))
            value = pii_len
            if pii_len > 1:
                value = math.log2(pii_len)
            train_data[PATTERN_MATCHES].append(value)
            index += 1
            print(
                    "\rProcessessing train: "
                    + "("
                    + str(index)
                    + "/"
                    + str(int(convos_len))
                    + ")",
                    end="",
                    flush=True,
                )

def score_accuracy(data, data_labels):
    """Determine the accuracy of the passed in data

    Args:
        data (dictionary {# matched PII, requested PII score}): The dictionary of data taken from the train dataset
        data_labels (array of {index, message, label}): Array of dictionaries of the test data
    """
    labels = []
    for i in range(len(data[REQUESTED_SCORE])):
        if data[PATTERN_MATCHES][i] > 0:
            labels.append(1)
        else:
            labels.append(0)

    num_correct = 0
    index =0
    for i in labels:
        if labels[index] > 0 and data_labels[index]["label"] != "None":
            num_correct += 1
        index +=1
    print("\nAccuracy: " + str((num_correct / len(labels))))

def db_scan(data):
    """Perform DBSCAN on the passed in data. Use clustering to visualize if PII is present in a converssation

    Args:
        data (dictionary): Dictionary of data
    """
    formatted = []
    for i in range(len(data[REQUESTED_SCORE])):
        if ENHANCED:
            formatted.append(np.array([data[REQUESTED_SCORE][i], data[PATTERN_MATCHES][i]]))
        else:
            formatted.append(np.array([data[PATTERN_MATCHES][i], 0]))
       
    X = np.array(formatted)
    epsilon = 0.01
    if ENHANCED:
        epsilon = find_epsilon(X)
    dbscan_cluster1 = DBSCAN(eps=epsilon, min_samples=8)
    dbscan_cluster1.fit(X)

    # Visualizing DBSCAN
    plt.scatter(X[:, 0], 
                X[:, 1], 
                c=dbscan_cluster1.labels_)
    if ENHANCED:
            plt.xlabel(REQUESTED_SCORE)
            plt.ylabel(PATTERN_MATCHES)
    else:
        plt.xlabel(PATTERN_MATCHES)
        plt.ylabel(REQUESTED_SCORE)

    # Number of Clusters
    labels=dbscan_cluster1.labels_
    n_clus=len(set(labels))-(1 if -1 in labels else 0)
    print('\nEstimated no. of clusters: %d' % n_clus)

    # Identify Noise
    n_noise = list(dbscan_cluster1.labels_).count(-1)
    print('Estimated no. of noise points: %d' % n_noise)
    plt.show()

def find_epsilon(X):
    """Use nearest neighbors and and the knee locator to determine the best epsilon value

    Args:
        X (np.array): The numpy array data

    Returns:
        float: epsilon
    """
    from sklearn.neighbors import NearestNeighbors

    nearest_neighbors = NearestNeighbors(n_neighbors=11)
    df=pd.DataFrame(X)
    df=df.rename(columns={0: "X1", 1:"X2"})
    neighbors = nearest_neighbors.fit(df)

    distances, indices = neighbors.kneighbors(df)
    distances = np.sort(distances[:,10], axis=0)

    if INCLUDE_PLOTS and ENHANCED:
        plt.plot(distances)
        plt.xlabel("Points")
        plt.ylabel("Distance")

    from kneed import KneeLocator

    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

    if INCLUDE_PLOTS and ENHANCED:
        knee.plot_knee()
        plt.xlabel("Points")
        plt.ylabel("Distance")
        print(distances[knee.knee])
        plt.show()

    return distances[knee.knee]
   

def main(run_type):
    """Run the passed in type of data

    Args:
        run_type (String): The kind of run to do
    """
    train_data = {
            REQUESTED_SCORE: [],
            PATTERN_MATCHES: []
        }
    test_data = {
            REQUESTED_SCORE: [],
            PATTERN_MATCHES: []
        }

    if run_type == "TRAIN":
        score_accuracy(train_data, process_train_data(train_data))
        # db_scan(train_data)
    elif run_type == "TEST":
        process_test_data(test_data)
        db_scan(test_data)

if __name__ == "__main__":
    main("TRAIN")
