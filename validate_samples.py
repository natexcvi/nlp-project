import json
import re


def validate_regex(data):
    print("Size of data: {}".format(len(data)))
    for description, details in data.items():
        pattern = details["regex"]
        for match in details["string_matches"]:
            if not re.fullmatch(pattern, match):
                print(f"Error: '{match}' should match pattern '{pattern}', but it does not.")
        for mismatch in details["string_mismatches"]:
            if re.fullmatch(pattern, mismatch):
                print(f"Error: '{mismatch}' should NOT match pattern '{pattern}', but it does.")


if __name__ == "__main__":
    with open("data/KB13/samples.json", "r") as file:
        data = json.load(file)
    validate_regex(data)
