import argparse
import json

import requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/relation/predict")
    args = parser.parse_args()

    payload = {
        "sentence": "Barack Obama was born in Honolulu.",
        "entity_1": "Barack Obama",
        "entity_2": "Honolulu",
        "relation_name": "place_of_birth",
        "relation_description": "entity_1 was born in entity_2",
    }

    response = requests.post(args.url, json=payload, timeout=120)
    print("status:", response.status_code)
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
