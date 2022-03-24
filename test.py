import functools
import json
import math
import re
import sys
from collections import defaultdict
from pydoc import doc

import matplotlib.pyplot as plt
import pandas as pd

import process_conversation as pc

document_filenames = {0 : "convo1.txt", # 82
                      1 : "convo2.txt", # 218
                      2 : "convo3.txt", # 36
                      3 : "convo4.txt", # 14
                      4 : "convo5.txt", # 13
                      5 : "convo6.txt", # 375
                      6 : "convo7.txt", # 39
                      7 : "convo8.txt", # 4
                      8 : "convo9.txt"} # 3

patterns = {"ADDRESS_PATTERN": re.compile(r"(street|road|drive|lane|avenue|boulevard|highway|township|north|south|east|west)"),
            "US_PHONE_PATTERN": re.compile(r"(?i)((\+?1(\.|-|\s)?)?)\s*((\(?\d{3}\)?(\.|-|\s*)?)?)\s*(\d{3}(\.|-|\s*)?)\s*(\d{4}\s*(((x|ext)\.?(ension)?)\s*\d*)?)"),
            "EMAIL_PATTERN": re.compile(r"([\w\.-]+)@([\da-zA-Z\.-]+)\.([a-zA-Z\.]{2,6})"),
            "SSN_PATTERN": re.compile(r"(?!000|666)[0-8]\d{2}(-|\s)(?!00)\d{2}(-|\s)(?!0000)\d{4}"),
            "VISA_PATTERN": re.compile(r"4[0-9]{12}(?:[0-9]{3})?"),
            "MC_PATTERN": re.compile(r"(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}"),
            "AMEX_PATTERN": re.compile(r"3[47][0-9]{13}"),
            "DISCOVER_PATTERN": re.compile(r"6(?:011|5[0-9]{2})[0-9]{12}")}

def main():
    with open("Z:\STIR\ML\IGDDPIIShare\MySQL Exports\data_10.json", encoding="utf-8") as fh:
        content = json.load(fh, strict=False)
        convos = {}

        for i in content:
            if i["ConversationID"] not in convos:
                convos[i["ConversationID"]] = []
            # elif len(convos) > 1000:
            #     break
            convos[i["ConversationID"]].append(i["Message"])

        data = {
            "PII Requested Score": [],
            "PII Pattern Matches": []
        }
        index = 1
        convos_len = len(convos)
        print("Processessing convos: (0/" + str(int(convos_len)) + ")", end="")
        for i in convos:
            conversation = "\n".join(convos[i])
            data["PII Requested Score"].append(pc.process_conversation(conversation))
            pii_len = 0
            for j in patterns:
                pii_len += len(patterns[j].findall(conversation))
            data["PII Pattern Matches"].append(pii_len)
            # print(str(index))
            index += 1
            print(
                    "\rProcessessing convos: "
                    + "("
                    + str(index)
                    + "/"
                    + str(int(convos_len))
                    + ")" + str(conversation.count('\n')),
                    end="",
                    flush=True,
                )

        df = pd.DataFrame(data,columns=['PII Requested Score','PII Pattern Matches'])
        print("\n")
        print (df)
        df.plot(x ='PII Requested Score', y='PII Pattern Matches', kind = 'scatter')
        plt.show()

if __name__ == "__main__":
    main()
