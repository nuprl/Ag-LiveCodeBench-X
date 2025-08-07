#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets==3.6.*",
# ]
# [tool.uv]
# exclude-newer = "2025-07-08T00:00:00Z"
# ///
"""
This script builds the LiveCodeBench-X dataset and stores it to disk. Note
that LiveCodeBench requires downloading nearly 3GB of data, and also requires
more than 64GB RAM to process the data.
"""

import datasets
import argparse
import json
import pickle
import zlib
import base64


def load_lcb_private_tests(item):
    """
    LiveCodeBench compresses its private tests because they are enormous (8GB
    when we write our 499 problem subset to disk).
    """
    loaded = json.loads(
        pickle.loads(
            zlib.decompress(
                base64.b64decode(item["private_test_cases"].encode("utf-8"))
            )
        )
    )
    return {"private_test_cases": loaded}


def all_stdin_tests(item):
    """
    We only keep the problems that have all stdin tests. The others require a specific programming language.
    """
    return all(test["testtype"] == "stdin" for test in item["private_test_cases"])


def prepare_lcb(version_tag: str, num_proc: int) -> datasets.Dataset:
    lcb = datasets.load_dataset(
        "livecodebench/code_generation_lite",
        version_tag=version_tag,
        trust_remote_code=True,
        split="test",
    )
    cleaned_lcb = lcb.map(load_lcb_private_tests, num_proc=num_proc)
    cleaned_lcb = cleaned_lcb.select_columns(
        ["question_id", "question_content", "private_test_cases"]
    )
    filtered_lcb = cleaned_lcb.filter(all_stdin_tests, num_proc=num_proc)
    print(
        f"LiveCodeBench {version_tag} has {len(cleaned_lcb)} problems, we support {len(filtered_lcb)} problems."
    )
    return filtered_lcb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-tag", type=str, default="release_v5")
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--output-path", type=str, default="./lcbx.jsonl")
    args = parser.parse_args()
    dataset = prepare_lcb(args.version_tag, args.num_proc)
    dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
