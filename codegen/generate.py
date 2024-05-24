import os
from os import PathLike
from typing import List
import json

from model import DecoderBase, make_model
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


def construct_contract_prompt(prompt: str, contract_type: str, contract: str) -> str:
    if contract_type == "none":
        return prompt
    elif contract_type == "docstring":
        # embed within the docstring
        sep = ""
        if '"""' in prompt:
            sep = '"""'
        elif "'''" in prompt:
            sep = "'''"
        assert sep != ""
        l = prompt.split(sep)
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        l[1] = (
            l[1] + contract + "\n" + " " * (len(contract) - len(contract.lstrip()) - 1)
        )
        return sep.join(l)
    elif contract_type == "code":
        # at the beginning of the function
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        return prompt + contract


def codegen(
    workdir: PathLike,
    model: DecoderBase,
    dataset: str,
    greedy=False,
    n_samples=1,
    id_range=None,
    version="default",
    resume=True,
    use_token_healing: bool = False,
    n_char_back=0,
    token_healing_sample_constrained: bool = False,
    token_healing_sample_predictions: bool = False,
):
    with Progress(
        TextColumn(f"{dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        if dataset == "humaneval":
            from evalplus.data import get_human_eval_plus

            dataset = get_human_eval_plus(version=version)
        elif dataset == "mbpp":
            from evalplus.data import get_mbpp_plus

            dataset = get_mbpp_plus(version=version)

        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            p_name = task_id.replace("/", "_")
            os.makedirs(os.path.join(workdir, p_name), exist_ok=True)
            log = f"Codegen: {p_name} @ {model}"
            n_existing = 0
            if resume:
                # count existing .py files
                n_existing = len(
                    [
                        f
                        for f in os.listdir(os.path.join(workdir, p_name))
                        if f.endswith(".py")
                    ]
                )
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"

            nsamples = n_samples - n_existing
            p.console.print(log)

            sidx = n_samples - nsamples
            while sidx < n_samples:
                outputs = model.codegen(
                    task["prompt"],
                    do_sample=not greedy,
                    num_samples=n_samples - sidx,
                    use_token_healing=use_token_healing,
                    n_char_back=n_char_back,
                    token_healing_sample_constrained=token_healing_sample_constrained,
                    token_healing_sample_predictions=token_healing_sample_predictions,
                )
                assert outputs, "No outputs from model!"
                if isinstance(outputs, dict):
                    # save to json
                    with open(
                        os.path.join(workdir, p_name, f"{sidx}.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(impl, f)
                    # save preds to .py
                    outputs = outputs["predictions"]

                if isinstance(outputs, list):
                    for impl in outputs:
                        try:
                            with open(
                                os.path.join(workdir, p_name, f"{sidx}.py"),
                                "w",
                                encoding="utf-8",
                            ) as f:
                                if model.is_direct_completion():
                                    f.write(task["prompt"] + impl)
                                else:
                                    f.write(impl)
                        except UnicodeEncodeError:
                            continue
                sidx += 1


def main(
    model: str,
    dataset: str,
    root: str,
    bs: int = 1,
    n_samples: int = 1,
    temperature: float = 0.0,
    resume: bool = True,
    greedy: bool = False,
    id_range: List = None,
    version: str = "default",
    backend: str = "hf",
    base_url: str = None,
    use_token_healing: bool = False,
    n_char_back: int = 0,
    token_healing_sample_constrained: bool = False,
    token_healing_sample_predictions: bool = False,
    tp: int = 1,
):
    assert dataset in ["humaneval", "mbpp"], f"Invalid dataset {dataset}"
    assert backend in ["vllm", "hf", "openai"]

    if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
        temperature = 0
        bs = 1
        n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    # Make project dir
    os.makedirs(root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(root, dataset), exist_ok=True)
    # Make dir for codes generated by each model
    model_runner = make_model(
        model=model,
        backend=backend,
        batch_size=bs,
        temperature=temperature,
        dataset=dataset,
        base_url=base_url,
        tp=tp,
    )
    identifier = (
        model.replace("/", "--")
        + f"_{backend}_temp_{temperature}_th_{use_token_healing}_thsc_{token_healing_sample_constrained}_thsp_{token_healing_sample_predictions}_fix"
    )
    if n_char_back > 0:
        identifier += f"_ncb_{n_char_back}"

    workdir = os.path.join(root, dataset, identifier)
    os.makedirs(workdir, exist_ok=True)
    codegen(
        workdir=workdir,
        dataset=dataset,
        greedy=greedy,
        model=model_runner,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
        version=version,
        use_token_healing=use_token_healing,
        n_char_back=n_char_back,
        token_healing_sample_constrained=token_healing_sample_constrained,
        token_healing_sample_predictions=token_healing_sample_predictions,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
