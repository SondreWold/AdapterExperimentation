from os import sep
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import requests
import numpy as np
import csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Creates a dictionary corpus where they key is a concept from CN and the value is its corresponding description on wikidata")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="The name of the relational file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path",
    )

    return parser.parse_args()


def request_description(concept: str) -> str:
    url = "https://www.wikidata.org/w/api.php"

    querystring = {"action": "wbgetentities", "format": "json", "sites": "enwiki",
                   "titles": concept, "props": "descriptions", "languages": "en"}

    payload = ""
    headers = {
        'cache-control': "no-cache",
    }

    description = requests.request(
        "GET", url, data=payload, headers=headers, params=querystring).json()

    if description["success"] != 1:
        print("Failed to fetch")
        return ""

    entity_keys = list(description["entities"].keys())
    entity = entity_keys[0]
    if entity == "-1":
        print("No match found in wikidata")
        return "Wikimedia disambiguation page"

    try:
        value = description["entities"][entity]["descriptions"]["en"]["value"]
    except KeyError:
        print("Failed to locate description within json object")
        print(description)
        value = "Wikimedia disambiguation page"
    return value


def main(args):
    SEED = 42
    np.random.seed(SEED)

    res = {}

    with open(args.input) as f:
        c = 0
        while c < 3:
            c += 1
            line = f.readline()
            if not line:
                break

            c1, c2 = line.split("\t")
            c1, c2 = c1[0].upper() + c1[1:], c2[0].upper() + c2[1:]

            print(f"Concept 1. {c1}, Concept 2. {c2}")

            if len(c1) <= 1 or len(c2) <= 1:
                continue

            c1_desc = request_description(c1)
            c2_desc = request_description(c2)

            if c1_desc != "Wikimedia disambiguation page":
                res[c1] = c1_desc

            if c2_desc != "Wikimedia disambiguation page":
                res[c2] = c2_desc

    with open(args.output, "w", encoding='utf-8', newline='') as output:
        for concept, desc in res.items():
            line = str(concept).strip() + "\t" + str(desc).strip() + "\n"
            output.write(line)


if __name__ == "__main__":
    main(parse_args())
