import json
import csv
import json2csv

with open("data/analysis/someMatches.json", "r") as in_file:
    matches = json.load(in_file)
    with open("data/analysis/someMatches.csv", "w") as out_file:
        csv_writer = csv.writer(out_file)
        for k1 in matches.keys():
            for i, k2 in matches[k1].keys():

                if k2.startswith("chat"):
                    for i in len()

