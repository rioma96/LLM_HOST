import argparse
import json

import requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    args = parser.parse_args()

    payload = {
        "mode": "review",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Items:\n"
                    '[{"sample_id":"s1","text":"Barack Obama was born in Honolulu.",'
                    '"head":"Barack Obama","tail":"Honolulu",'
                    '"candidate_labels":["/people/person/place_of_birth"]}]'
                ),
            }
        ],
    }

    response = requests.post(args.url, json=payload, timeout=120)
    print("status:", response.status_code)
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
