import gzip
import json
import os
import pathlib
from typing import Dict, List

import tempdir
import wget
from appdirs import user_cache_dir

HUMANEVAL_PLUS_PATH = pathlib.Path(__file__).parent.parent / "HumanEvalPlus.jsonl"
CACHE_DIR = user_cache_dir("eval-plus")


def get_human_eval_plus() -> List[Dict[str, str]]:
    """Get HumanEvalPlus locally.
    Returns:
        List[Dict[str, str]]: List of dicts with keys "task_id", "prompt", "contract", "isignature", "docstring", "canonical_solution", "base_input", "fuzz_input"
    Notes:
        "task_id" is the identifier string for the task.
        "prompt" is the function signature with docstring.
        "contract" is the assertions for the function's input (validity).
        "isignature" is the function's input signature.
        "docstring" is the docstring.
        "canonical_solution" is the ground-truth implementation for diff-testing.
        "base_input" is the test inputs.
    """
    human_eval_plus = open(HUMANEVAL_PLUS_PATH, "r").read()
    human_eval_plus = human_eval_plus.split("\n")
    human_eval_plus = [json.loads(line) for line in human_eval_plus if line]
    return human_eval_plus


def get_human_eval() -> List[Dict[str, str]]:
    """Get HumanEval from OpenAI's github repo and return as a list of parsed dicts.

    Returns:
        List[Dict[str, str]]: List of dicts with keys "prompt", "test", "entry_point"

    Notes:
        "task_id" is the identifier string for the task.
        "prompt" is the prompt to be used for the task (function signature with docstrings).
        "test" is test-cases wrapped in a `check` function.
        "entry_point" is the name of the function.
    """
    # Check if human eval file exists in CACHE_DIR
    human_eval_path = os.path.join(CACHE_DIR, "HumanEval.jsonl")

    human_eval = None
    if not os.path.exists(human_eval_path):
        # Install HumanEval dataset and parse as jsonl
        # https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz
        print("Downloading HumanEval dataset...")
        with tempdir.TempDir() as tmpdir:
            human_eval_gz_path = os.path.join(tmpdir, "HumanEval.jsonl.gz")
            wget.download(
                "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
                human_eval_gz_path,
            )

            with gzip.open(human_eval_gz_path, "rb") as f:
                human_eval = f.read().decode("utf-8")

        # create CACHE_DIR if not exists
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        # Write the original human eval file to CACHE_DIR
        with open(human_eval_path, "w") as f:
            f.write(human_eval)

    human_eval = open(human_eval_path, "r").read() if not human_eval else human_eval
    human_eval = human_eval.split("\n")
    human_eval = [json.loads(line) for line in human_eval if line]

    # Handle 115_max_fill.py to make its docstring well-formed
    human_eval[114]["prompt"] = "import math\n" + human_eval[114]["prompt"].replace(
        "import math\n", ""
    )

    return human_eval
