"""CodeFeedback training data and EvalPlus MBPP+ evaluation utilities."""

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset


ALPACA_CODE_TRAIN_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


_CODE_DATA_CACHE = {}


def _first_code_block(answer):
    """Keep the answer through its first fenced block, as in LoRA-One."""
    if not isinstance(answer, str) or "```" not in answer:
        return None
    return "```".join(answer.split("```")[:2]) + "```"


def _tokenize_code_batch(batch, tokenizer, max_length, train_on_inputs):
    normalized = []
    for query, answer in zip(batch["query"], batch["answer"]):
        output = _first_code_block(answer)
        if not isinstance(query, str) or not query.strip() or output is None:
            continue
        point = {"instruction": query, "input": "", "output": output}
        normalized.append(point)

    if not normalized:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    user_prompts = [
        ALPACA_CODE_TRAIN_TEMPLATE.format(instruction=point["instruction"])
        for point in normalized
    ]
    full_prompts = [
        user_prompt + " " + point["output"]
        for user_prompt, point in zip(user_prompts, normalized)
    ]
    full_tokens = tokenizer(
        full_prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    user_tokens = tokenizer(
        user_prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )

    result = {"input_ids": [], "attention_mask": [], "labels": []}
    eos_id = tokenizer.eos_token_id
    for input_ids, attention_mask, user_ids in zip(
        full_tokens["input_ids"],
        full_tokens["attention_mask"],
        user_tokens["input_ids"],
    ):
        # LoRA-One discards samples at or beyond the sequence limit instead
        # of silently training on truncated code.
        if len(input_ids) >= max_length:
            continue
        input_ids = list(input_ids)
        attention_mask = list(attention_mask)
        if eos_id is not None and (not input_ids or input_ids[-1] != eos_id):
            input_ids.append(eos_id)
            attention_mask.append(1)
        labels = list(input_ids)
        if not train_on_inputs:
            prompt_length = min(len(user_ids), len(labels))
            labels[:prompt_length] = [-100] * prompt_length
        result["input_ids"].append(input_ids)
        result["attention_mask"].append(attention_mask)
        result["labels"].append(labels)
    return result


def load_codefeedback_train(
    data_path,
    tokenizer,
    max_length=1024,
    train_on_inputs=False,
    val_set_size=0,
    seed=42,
    max_train_samples=100000,
):
    """Load and tokenize local CodeFeedback data for causal-LM training."""
    max_length = int(max_length)
    val_set_size = int(val_set_size)
    max_train_samples = (
        None if max_train_samples is None else int(max_train_samples)
    )
    if max_length <= 1 or val_set_size < 0:
        raise ValueError("max_length must exceed 1 and val_set_size must be non-negative.")
    if max_train_samples is not None and max_train_samples <= 0:
        raise ValueError("max_train_samples must be positive or null.")

    cache_key = (
        os.path.abspath(data_path),
        getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__),
        len(tokenizer),
        max_length,
        bool(train_on_inputs),
        val_set_size,
        max_train_samples,
        int(seed),
    )
    if cache_key in _CODE_DATA_CACHE:
        return _CODE_DATA_CACHE[cache_key]

    raw = load_dataset("json", data_files=data_path, split="train")
    required = {"query", "answer"}
    if not required.issubset(raw.column_names):
        raise ValueError(
            f"CodeFeedback must contain {sorted(required)}, got {raw.column_names}."
        )
    # Dataset.shuffle is not in-place.  Assigning its result fixes a subtle
    # reproducibility issue in the reference LoRA-One loader.
    raw = raw.shuffle(seed=int(seed))
    encoded = raw.map(
        lambda batch: _tokenize_code_batch(
            batch, tokenizer, max_length, bool(train_on_inputs)
        ),
        batched=True,
        batch_size=256,
        remove_columns=raw.column_names,
        desc="Tokenizing CodeFeedback",
    )

    available = len(encoded)
    train_capacity = max(0, available - val_set_size)
    train_count = (
        train_capacity
        if max_train_samples is None
        else min(max_train_samples, train_capacity)
    )
    val_count = min(val_set_size, max(0, available - train_count))
    print(
        f"[CodeFeedback] raw={len(raw)}, valid={available}, "
        f"train={train_count}, validation={val_count}"
    )
    train_data = encoded.select(range(train_count))
    val_data = (
        encoded.select(range(train_count, train_count + val_count))
        if val_count
        else None
    )
    result = (train_data, val_data)
    _CODE_DATA_CACHE[cache_key] = result
    return result


def _archive_previous_result(path):
    """Keep the previous EvalPlus result so a rerun cannot reuse stale scores."""
    if not path.exists():
        return
    backup = path.with_name(path.name + ".bak")
    while backup.exists():
        backup = backup.with_name(backup.name + ".bak")
    path.rename(backup)


def _summarize_mbpp_results(result_path, expected_task_ids):
    """Count base and base-plus-extra passes from one result per MBPP+ task."""
    with result_path.open(encoding="utf-8") as result_file:
        results = json.load(result_file)["eval"]
    if set(results) != set(expected_task_ids):
        raise ValueError(
            f"EvalPlus scored {len(results)} tasks; expected {len(expected_task_ids)}."
        )
    if any(len(task_results) != 1 for task_results in results.values()):
        raise ValueError("Greedy Pass@1 requires exactly one result per task.")
    total = len(results)
    base_correct = sum(
        task_results[0]["base_status"] == "pass"
        for task_results in results.values()
    )
    plus_correct = sum(
        task_results[0]["base_status"] == "pass"
        and task_results[0]["plus_status"] == "pass"
        for task_results in results.values()
    )
    return {
        "dataset": "MBPP+",
        "total": total,
        "base_correct": base_correct,
        "base_pass_at_1": base_correct / total,
        "plus_correct": plus_correct,
        "plus_pass_at_1": plus_correct / total,
        "eval_results": str(result_path),
    }


def evaluate_mbpp_model(
    model,
    tokenizer,
    output_dir,
    max_new_tokens=768,
    parallel=None,
    version="default",
    force_base_prompt=True,
):
    """Generate once per MBPP+ task and score base and enhanced tests."""
    from evalplus.codegen import codegen as evalplus_codegen
    from evalplus.data import get_mbpp_plus
    from evalplus.evaluate import evaluate as evalplus_evaluate
    from evalplus.provider.base import DecoderBase
    from evalplus.provider.utility import (
        extra_eos_for_direct_completion,
        make_raw_chat_prompt,
    )
    from stop_sequencer import StopSequencer

    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if parallel is not None:
        parallel = int(parallel)
        if parallel <= 0:
            raise ValueError("parallel must be positive.")

    instruction_prefix = (
        "Please provide a self-contained Python script that solves the following "
        "problem in a markdown code block:"
    )
    response_prefix = (
        "Below is a Python script with a self-contained function that solves the "
        "problem and passes corresponding tests:"
    )

    class LoadedModelDecoder(DecoderBase):
        def __init__(self):
            super().__init__(
                name=getattr(model.config, "_name_or_path", "loaded-model"),
                batch_size=1,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                instruction_prefix=instruction_prefix,
                response_prefix=response_prefix,
            )
            self.model = model
            self.tokenizer = tokenizer
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.eos = list(self.eos)
            if self.is_direct_completion():
                self.eos += extra_eos_for_direct_completion("mbpp")
            else:
                self.eos += ["\n```\n"]
            self.model.eval()

        def is_direct_completion(self):
            return force_base_prompt or self.tokenizer.chat_template is None

        @torch.inference_mode()
        def codegen(self, prompt, do_sample=False, num_samples=1):
            if do_sample or num_samples != 1:
                raise ValueError("Only greedy Pass@1 is supported.")
            model_prompt = (
                prompt if self.is_direct_completion() else make_raw_chat_prompt(
                    prompt, self.instruction_prefix, self.response_prefix,
                    self.tokenizer,
                )
            )
            input_device = self.model.get_input_embeddings().weight.device
            input_tokens = self.tokenizer.encode(
                model_prompt, return_tensors="pt"
            ).to(input_device)
            stopped_model = StopSequencer(
                self.model, model_type="causal", tokenizer=self.tokenizer
            ).register_stop_texts(
                stop_texts=self.eos, input_length=input_tokens.size(-1)
            )
            output_tokens = stopped_model.generate(
                input_tokens,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated = self.tokenizer.batch_decode(
                output_tokens[:, input_tokens.size(-1):],
                skip_special_tokens=True,
            )[0]
            positions = [generated.find(stop) for stop in self.eos]
            positions = [position for position in positions if position >= 0]
            if positions:
                generated = generated[:min(positions)]
            return [generated.replace("\t", "    ")]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "mbpp_evalplus_samples.jsonl"
    result_path = output_dir / "mbpp_evalplus_samples_eval_results.json"
    _archive_previous_result(result_path)
    for generated_path in (
        samples_path,
        output_dir / "mbpp_evalplus_samples.raw.jsonl",
    ):
        generated_path.unlink(missing_ok=True)

    evalplus_codegen(
        target_path=str(samples_path), model=LoadedModelDecoder(),
        dataset="mbpp", greedy=True, n_samples=1,
        version=version, resume=False,
    )
    evalplus_evaluate(
        dataset="mbpp", samples=str(samples_path), base_only=False,
        parallel=parallel, version=version,
    )
    summary = _summarize_mbpp_results(
        result_path, get_mbpp_plus(version=version)
    )
    summary["version"] = version
    summary["samples"] = str(samples_path)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    return summary


__all__ = ["evaluate_mbpp_model", "load_codefeedback_train"]
