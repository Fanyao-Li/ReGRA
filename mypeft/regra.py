import gc
import json
import os
import random
import socket
from datetime import datetime

import torch
import transformers
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from utils.dev_parse import print_dict_paths
from utils.evaluate_utils import COMMONSENSE_DATASETS, MATH_DATASETS, evaluate_dataset
from utils.finetune_utils import generate_and_tokenize_prompt, get_data
from utils.logger import get_log
from utils.train_utils import (
    convert_lora_params_dtype,
    convert_target_modules,
    get_trainable_params_numbers,
    print_delta_time,
    set_global_seed,
)



def _is_glue(config):
    return str(config.datamode).lower() == "glue"


def _is_commonsense(config):
    return str(config.datamode).lower() == "commonsense"


def _random_commonsense_calibration_records(config, total_samples):
    """Sample DNS calibration examples from the mixed training set."""
    total_samples = int(total_samples)
    if total_samples <= 0:
        raise ValueError("Commonsense DNS calibration sample count must be positive.")

    with open(config.data_path, "r", encoding="utf-8") as input_file:
        records = json.load(input_file)
    if not isinstance(records, list):
        raise ValueError("Commonsense training data must be a JSON list.")
    if total_samples > len(records):
        raise ValueError(
            "Not enough commonsense training examples for DNS calibration: "
            f"requested={total_samples}, available={len(records)}."
        )

    indices = random.Random(int(config.seed)).sample(
        range(len(records)), total_samples
    )
    return [records[index] for index in indices]


def _get_commonsense_calibration_data(config, tokenizer, total_samples):
    """Tokenize a random commonsense DNS calibration subset."""
    from datasets import Dataset

    train_config = config.train_config
    records = _random_commonsense_calibration_records(
        config, total_samples=total_samples
    )
    return Dataset.from_list(records).map(
        lambda sample: generate_and_tokenize_prompt(
            sample,
            tokenizer,
            train_config.max_length,
            train_config.train_on_inputs,
        )
    )


def _get_target_modules(config):
    target_modules = config.lora_config.target_modules
    if _is_glue(config):
        if isinstance(target_modules, str):
            return [
                name.strip() for name in target_modules.split(",")
                if name.strip()
            ]
        return list(target_modules)
    return convert_target_modules(target_modules)


def _prepare_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if _is_glue(config):
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "GLUE sequence classification requires a pad or EOS token."
                )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    else:
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
    return tokenizer


def _load_glue_data(config, tokenizer):
    from utils.glue_utils import load_glue_data

    train_config = config.train_config
    return load_glue_data(
        task_name=config.glue_task,
        tokenizer=tokenizer,
        max_length=train_config.max_length,
        dataset_name=getattr(config, "data_path", "nyu-mll/glue"),
        cache_dir=getattr(train_config, "glue_cache_dir", None),
        max_train_samples=getattr(
            train_config, "glue_max_train_samples", None
        ),
        max_eval_samples=getattr(
            train_config, "glue_max_eval_samples", None
        ),
        pad_to_max_length=bool(
            getattr(train_config, "glue_pad_to_max_length", False)
        ),
        validation_ratio=float(
            getattr(train_config, "glue_validation_ratio", 0.0)
        ),
        seed=config.seed,
    )


def _load_task_model(config, tokenizer, glue_data=None, device_map="auto"):
    if _is_glue(config):
        if glue_data is None:
            glue_data = _load_glue_data(config, tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model,
            num_labels=glue_data["num_labels"],
            problem_type=(
                "regression" if glue_data["is_regression"]
                else "single_label_classification"
            ),
            dtype=torch.float32,
            device_map=device_map,
            ignore_mismatched_sizes=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        return model
    return AutoModelForCausalLM.from_pretrained(
        config.model, torch_dtype=torch.bfloat16, device_map=device_map
    )


def _make_peft_config(config):
    lora_config = config.lora_config
    kwargs = dict(
        r=lora_config.rank,
        lora_alpha=lora_config.lora_alpha,
        target_modules=_get_target_modules(config),
        lora_dropout=lora_config.lora_dropout,
        use_dora=getattr(lora_config, "use_dora", False),
    )
    if _is_glue(config):
        kwargs.update(
            task_type=TaskType.SEQ_CLS,
            modules_to_save=["classifier", "score"],
        )
    return LoraConfig(**kwargs)


def _make_data_collator(config, tokenizer):
    if _is_glue(config):
        return transformers.DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        )
    return transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )


def _freeze_stage1_classification_head(model):
    """Freeze Stage-1's head while preserving gradients to the encoder."""
    frozen = []
    for name, parameter in model.named_parameters():
        parts = set(name.split("."))
        if "classifier" in parts or "score" in parts:
            parameter.requires_grad_(False)
            frozen.append(name)
    if not frozen:
        raise ValueError("Could not find the Stage-1 GLUE classification head.")
    return frozen

def _get_task_train_data(
    config, tokenizer, val_set_size=None, glue_data=None
):
    """Dispatch task-specific training and validation data."""
    train_config = config.train_config
    if val_set_size is None:
        val_set_size = train_config.val_set_size
    if _is_glue(config):
        if glue_data is None:
            glue_data = _load_glue_data(config, tokenizer)
        return glue_data["train_dataset"], glue_data["eval_dataset"]
    if str(config.datamode).lower() == "code":
        from utils.code_utils import load_codefeedback_train

        return load_codefeedback_train(
            data_path=config.data_path,
            tokenizer=tokenizer,
            max_length=train_config.max_length,
            train_on_inputs=train_config.train_on_inputs,
            val_set_size=val_set_size,
            seed=config.seed,
            max_train_samples=getattr(
                train_config, "code_max_train_samples", 100000
            ),
        )
    return get_data(
        config.data_path,
        val_set_size,
        tokenizer,
        train_config.max_length,
        train_config.train_on_inputs,
        seed=config.seed,
    )


def _evaluate_task(model, tokenizer, config, output_dir, logger):
    """Dispatch evaluation to GLUE, MBPP+, or legacy generation tasks."""
    test_config = config.test_config
    test_data = str(config.datamode).lower()
    set_global_seed(config.seed)
    if test_data == "glue":
        from utils.glue_utils import (
            evaluate_glue_trainer,
            make_glue_compute_metrics,
        )

        glue_data = _load_glue_data(config, tokenizer)
        evaluator = transformers.Trainer(
            model=model,
            args=transformers.TrainingArguments(
                output_dir=os.path.join(output_dir, "trainer_tmp"),
                per_device_eval_batch_size=int(test_config.test_batch_size),
                bf16=True,
                report_to=[],
            ),
            data_collator=_make_data_collator(config, tokenizer),
            compute_metrics=make_glue_compute_metrics(
                glue_data["task_name"], glue_data["is_regression"]
            ),
        )
        return evaluate_glue_trainer(
            evaluator, glue_data, output_dir, logger
        )
    if test_data == "code":
        from utils.code_utils import evaluate_mbpp_model

        summary = evaluate_mbpp_model(
            model=model,
            tokenizer=tokenizer,
            output_dir=os.path.join(output_dir, "mbpp"),
            max_new_tokens=int(getattr(test_config, "code_max_new_tokens", 768)),
            parallel=getattr(test_config, "code_eval_parallel", None),
            version=str(getattr(test_config, "mbpp_version", "default")),
            force_base_prompt=bool(getattr(test_config, "code_force_base_prompt", True)),
        )
        logger.info(
            "[MBPP+] base pass@1: %.6f (%d/%d); enhanced pass@1: %.6f (%d/%d)",
            summary["base_pass_at_1"], summary["base_correct"], summary["total"],
            summary["plus_pass_at_1"], summary["plus_correct"], summary["total"],
        )
        return summary["plus_pass_at_1"]

    test_datasets = MATH_DATASETS if test_data == "math" else COMMONSENSE_DATASETS
    dataset_ids = getattr(test_config, "test_dataset_ids", "")
    if dataset_ids != "":
        test_datasets = [test_datasets[int(i)] for i in dataset_ids]

    accuracy = 0.0
    for test_dataset in test_datasets:
        accuracy = evaluate_dataset(
            model, tokenizer, test_data, test_dataset,
            test_config.test_batch_size, output_dir,
        )
        if accuracy < 0.01:
            logger.info("stop evaluate due to poor accuracy.")
            break
    return accuracy


def _find_adapter_dir(stage_path):
    """Prefer the final adapter; otherwise fall back to the latest checkpoint."""
    if os.path.isfile(os.path.join(stage_path, "adapter_model.safetensors")):
        return stage_path

    checkpoints = sorted([
        d for d in os.listdir(stage_path)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(stage_path, d))
    ], key=lambda x: int(x.split("-")[1]))  # 按数字排序
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {stage_path}")
    
    return os.path.join(stage_path, checkpoints[-1])  # 返回最新的


def _merge_stage1(base_model, adapter_dir):
    """Merge Stage 1 so Stage 2 starts from exactly W0 + W1."""
    model = PeftModel.from_pretrained(
        base_model, adapter_dir, autocast_adapter_dtype=False
    )
    return model.merge_and_unload()


# ---------------------------------------------------------------------------
# Stage 1 – standard LoRA training, identical in spirit to run_lora().
# The adapter is saved so Stage 2 can load it.
# ---------------------------------------------------------------------------
def run_regra_stage1(config, config_dict):
    logger = get_log(config.output_dir, "setting")
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)

    server_name = socket.gethostname()
    logger.info(f"[ReGRA Stage 1] server_name: {server_name}")
    print_dict_paths(config_dict, logger)

    train_config = config.train_config
    test_config = config.test_config
    lora_config = config.lora_config
    is_glue = _is_glue(config)
    if getattr(lora_config, "use_dora", False):
        raise ValueError("ReGRA update orthogonality currently supports LoRA only, not DoRA.")
    set_global_seed(config.seed)
    ft_start_time = datetime.now()

    # ---- model ----
    tokenizer = _prepare_tokenizer(config)
    glue_data = _load_glue_data(config, tokenizer) if is_glue else None
    model = _load_task_model(config, tokenizer, glue_data, device_map="auto")

    # ---- LoRA ----
    peft_config = _make_peft_config(config)
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
    convert_lora_params_dtype(model, dtype=lora_config.dtype)

    os.makedirs(os.path.join(config.output_dir, "stage1"), exist_ok=True)

    logger.info(model)
    rate = get_trainable_params_numbers(
        model, path=os.path.join(config.output_dir, "stage1", "num_params.json")
    )
    logger.info(rate)

    logger.info("### Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: {param.dtype}")

    # ---- finetune ----
    if config.finetune:
        train_data, val_data = _get_task_train_data(
            config, tokenizer, glue_data=glue_data
        )
        has_validation = is_glue or train_config.val_set_size > 0
        best_metric = None
        if is_glue:
            glue_task = glue_data["task_name"]
            if glue_task in {"stsb", "mrpc", "qqp"}:
                best_metric = "combined_score"
            elif glue_task == "cola":
                best_metric = "matthews_correlation"
            else:
                best_metric = "accuracy"

        training_args = transformers.TrainingArguments(
            warmup_steps=100,
            save_only_model=True,
            per_device_train_batch_size=train_config.micro_batch_size,
            per_device_eval_batch_size=train_config.micro_batch_size,
            gradient_accumulation_steps=train_config.batch_size // train_config.micro_batch_size,
            gradient_checkpointing=train_config.use_gradient_checkpointing,
            num_train_epochs=train_config.num_epochs,
            learning_rate=train_config.lr,
            weight_decay=train_config.weight_decay,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps" if has_validation else "no",
            save_strategy="steps",
            eval_steps=train_config.eval_step if has_validation else None,
            save_steps=train_config.save_step,
            output_dir=os.path.join(config.output_dir, "stage1"),
            save_total_limit=1,
            load_best_model_at_end=has_validation,
            metric_for_best_model=best_metric,
            greater_is_better=True if is_glue else None,
            report_to=[],
            seed=config.seed,
        )
        trainer_kwargs = {}
        if is_glue:
            from utils.glue_utils import make_glue_compute_metrics

            trainer_kwargs["compute_metrics"] = make_glue_compute_metrics(
                glue_data["task_name"], glue_data["is_regression"]
            )
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            data_collator=_make_data_collator(config, tokenizer),
            **trainer_kwargs,
        )
        trainer.train()

        # Save Stage 1 adapter under output_dir/stage1/
        stage1_save_path = os.path.join(config.output_dir, "stage1")
        model.save_pretrained(stage1_save_path)
        logger.info(f"[ReGRA Stage 1] Adapter saved to {stage1_save_path}")
        print_delta_time(ft_start_time, logger)

    # ---- evaluate ----
    if config.evaluate:
        model.eval()
        if test_config.merge:
            model = model.merge_and_unload()
        torch.cuda.empty_cache()
        eval_start_time = datetime.now()
        _evaluate_task(
            model, tokenizer, config,
            os.path.join(config.output_dir, "stage1", "evaluated_result"),
            logger,
        )
        print_delta_time(eval_start_time, logger)


# ---------------------------------------------------------------------------
# Stage 2: learn W2 on the fixed base W0 + W1.
# ---------------------------------------------------------------------------
def run_regra_stage2(config, config_dict, residual_svd):
    """Train W2 on the fixed base W0 + W1."""
    regra_config = config.regra_config
    stage2_lr = float(regra_config.stage2_lr)
    if not torch.isfinite(torch.tensor(stage2_lr)) or stage2_lr <= 0:
        raise ValueError("regra_config.stage2_lr must be finite and positive.")
    # Auto-derive Stage 1 path from output_dir
    stage1_lora_path = os.path.join(config.output_dir, "stage1")

    orth_lambda = getattr(regra_config, "orth_lambda", 0.1)
    orth_eps = getattr(regra_config, "orth_eps", 1e-12)

    logger = get_log(config.output_dir, "setting")
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)

    server_name = socket.gethostname()
    logger.info(f"[ReGRA Stage 2] server_name: {server_name}")
    logger.info(f"[ReGRA Stage 2] stage1_lora_path: {stage1_lora_path}")
    logger.info(f"[ReGRA Stage 2] orth_lambda: {orth_lambda}")
    logger.info(f"[ReGRA Stage 2] orth_eps: {orth_eps}")
    logger.info(f"[ReGRA Stage 2] learning_rate: {stage2_lr}")
    logger.info("[ReGRA Stage 2] base initialization: W0 + W1")
    print_dict_paths(config_dict, logger)

    train_config = config.train_config
    lora_config = config.lora_config
    is_glue = _is_glue(config)
    if getattr(lora_config, "use_dora", False):
        raise ValueError("ReGRA update orthogonality currently supports LoRA only, not DoRA.")
    set_global_seed(config.seed)
    ft_start_time = datetime.now()

    # ---- tokenizer and task model ----
    tokenizer = _prepare_tokenizer(config)
    glue_data = _load_glue_data(config, tokenizer) if is_glue else None

    # ----------------------------------------------------------------
    # Step 1 and 2: load W0, then merge Stage 1 (including its GLUE head).
    # ----------------------------------------------------------------
    logger.info("[ReGRA Stage 2] Loading clean base model …")
    base_model = _load_task_model(
        config, tokenizer, glue_data, device_map="cuda"
    )

    logger.info("[ReGRA Stage 2] Loading Stage 1 adapter weights …")
    stage1_adapter_dir = _find_adapter_dir(stage1_lora_path)
    logger.info(f"[ReGRA Stage 2]   found at: {stage1_adapter_dir}")

    # Load adapter config for logging
    with open(os.path.join(stage1_adapter_dir, "adapter_config.json"), "r") as f:
        stage1_adapter_config = json.load(f)
    stage1_r = stage1_adapter_config["r"]
    stage1_lora_alpha = stage1_adapter_config["lora_alpha"]
    stage1_scale = stage1_lora_alpha / stage1_r
    logger.info(f"[ReGRA Stage 2] Stage 1 r={stage1_r}, alpha={stage1_lora_alpha}, scale={stage1_scale:.4f}")

    # Keep a CPU copy of LoRA weights for orth loss later.
    stage1_weights = load_safetensors(
        os.path.join(stage1_adapter_dir, "adapter_model.safetensors")
    )

    stage1_lora_sd = {}
    for k, v in stage1_weights.items():
        if "lora_" in k:
            new_k = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
            stage1_lora_sd[new_k] = v.cpu()
    del stage1_weights

    # ----------------------------------------------------------------
    # Step 3: W_new = W0 + W1.
    # ----------------------------------------------------------------
    logger.info("[ReGRA Stage 2] Merging Stage 1: W = W0 + W1 ...")
    base_model = _merge_stage1(base_model, stage1_adapter_dir)
    torch.cuda.empty_cache()
    logger.info("[ReGRA Stage 2] W0 + W1 merge complete.")

    # ----------------------------------------------------------------
    # Step 4: apply a fresh Stage 2 LoRA adapter to W_new.
    # ----------------------------------------------------------------
    logger.info("[ReGRA Stage 2] Applying Stage 2 LoRA adapter …")
    peft_config = _make_peft_config(config)
    model = get_peft_model(base_model, peft_config, autocast_adapter_dtype=False)
    if is_glue:
        frozen_head = _freeze_stage1_classification_head(model)
        logger.info(
            "[ReGRA Stage 2] Reusing frozen Stage-1 classification head: "
            f"parameters={len(frozen_head)}"
        )
    convert_lora_params_dtype(model, dtype=lora_config.dtype)

    os.makedirs(os.path.join(config.output_dir, "stage2"), exist_ok=True)

    logger.info(model)
    rate = get_trainable_params_numbers(
        model, path=os.path.join(config.output_dir, "stage2", "num_params.json")
    )
    logger.info(rate)

    logger.info("### Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: {param.dtype}")

    # ----------------------------------------------------------------
    # Step 5: finetune with W1/W2 Frobenius orthogonality.
    # ----------------------------------------------------------------
    if config.finetune:
        train_data, val_data = _get_task_train_data(
            config, tokenizer, glue_data=glue_data
        )
        has_validation = is_glue or train_config.val_set_size > 0
        best_metric = None
        if is_glue:
            glue_task = glue_data["task_name"]
            if glue_task in {"stsb", "mrpc", "qqp"}:
                best_metric = "combined_score"
            elif glue_task == "cola":
                best_metric = "matthews_correlation"
            else:
                best_metric = "accuracy"

        training_args = transformers.TrainingArguments(
            warmup_steps=100,
            save_only_model=True,
            per_device_train_batch_size=train_config.micro_batch_size,
            per_device_eval_batch_size=train_config.micro_batch_size,
            gradient_accumulation_steps=train_config.batch_size // train_config.micro_batch_size,
            gradient_checkpointing=train_config.use_gradient_checkpointing,
            num_train_epochs=train_config.num_epochs,
            learning_rate=stage2_lr,
            weight_decay=train_config.weight_decay,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps" if has_validation else "no",
            save_strategy="steps",
            eval_steps=train_config.eval_step if has_validation else None,
            save_steps=train_config.save_step,
            output_dir=os.path.join(config.output_dir, "stage2"),
            save_total_limit=1,
            load_best_model_at_end=has_validation,
            metric_for_best_model=best_metric,
            greater_is_better=True if is_glue else None,
            report_to=[],
            seed=config.seed,
        )
        trainer_kwargs = {}
        if is_glue:
            from utils.glue_utils import make_glue_compute_metrics

            trainer_kwargs["compute_metrics"] = make_glue_compute_metrics(
                glue_data["task_name"], glue_data["is_regression"]
            )
        trainer = ReGRAStage2Trainer(
            stage1_state_dict=stage1_lora_sd,
            orth_lambda=orth_lambda,
            orth_eps=orth_eps,
            residual_svd=residual_svd,
            projection_mode="dns",
            orth_warmup_steps=int(
                getattr(regra_config, "orth_warmup_steps", 0)
            ),
            basis_eps=float(getattr(regra_config, "basis_eps", 1e-7)),
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            data_collator=_make_data_collator(config, tokenizer),
            **trainer_kwargs,
        )
        trainer.train()
        model.save_pretrained(os.path.join(config.output_dir, "stage2"))
        logger.info(f"[ReGRA Stage 2] Adapter saved to {os.path.join(config.output_dir, 'stage2')}")
        print_delta_time(ft_start_time, logger)


# ---------------------------------------------------------------------------
# Evaluate the final W0 + W1 + W2 composition.

# ---------------------------------------------------------------------------
def run_regra_eval(config):
    """Evaluate only the fused composition W0 + W1 + W2."""
    if not config.evaluate:
        return

    logger = get_log(config.output_dir, "setting")
    test_config = config.test_config

    stage1_path = os.path.join(config.output_dir, "stage1")
    stage2_path = os.path.join(config.output_dir, "stage2")

    stage1_path = _find_adapter_dir(stage1_path)
    stage2_path = _find_adapter_dir(stage2_path)
    logger.info(f"[ReGRA Eval] stage1: {stage1_path}")
    logger.info(f"[ReGRA Eval] stage2: {stage2_path}")
    logger.info("[ReGRA Eval] composition: W0 + W1 + W2")

    logger.info("[ReGRA Fused Eval] Loading base model …")
    tokenizer = _prepare_tokenizer(config)
    glue_data = _load_glue_data(config, tokenizer) if _is_glue(config) else None
    model = _load_task_model(
        config, tokenizer, glue_data, device_map="auto"
    )

    # ----------------------------------------------------------------
# Step 1: W0 + W1.
    # ----------------------------------------------------------------
    logger.info("[ReGRA Fused Eval] W = W0 + W1 ...")
    model = _merge_stage1(model, stage1_path)
    torch.cuda.empty_cache()
    logger.info("[ReGRA Fused Eval] W0 + W1 merge complete.")

    # ----------------------------------------------------------------
    # Step 2: (W0 + W1) + W2.
    # ----------------------------------------------------------------
    logger.info("[ReGRA Fused Eval] W = W + ΔW₂ (PeftModel fusion) …")
    peft_s2 = PeftModel.from_pretrained(
        model,
        stage2_path,
        autocast_adapter_dtype=False,
    )
    model = peft_s2.merge_and_unload()
    torch.cuda.empty_cache()
    logger.info("[ReGRA Fused Eval] W0 + W1 + W2 complete.")

    # Evaluate
    model.eval()
    eval_start_time = datetime.now()
    eval_save_dir = os.path.join(
        config.output_dir, "fused", "w0_stage1_stage2", "evaluated_result"
    )
    _evaluate_task(model, tokenizer, config, eval_save_dir, logger)
    print_delta_time(eval_start_time, logger)


# ---------------------------------------------------------------------------
# Public entry point – dispatches to Stage 1 or Stage 2 based on config.
# ---------------------------------------------------------------------------

def _load_stage1_lora_state(adapter_dir):
    """Load Stage-1 LoRA tensors using PEFT ``named_parameters`` names."""
    weights = load_safetensors(os.path.join(adapter_dir, "adapter_model.safetensors"))
    state = {}
    for name, tensor in weights.items():
        if "lora_" not in name:
            continue
        name = name.replace(
            ".lora_A.weight", ".lora_A.default.weight"
        ).replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        state[name] = tensor.detach().cpu()
    return state


@torch.no_grad()

def _compress_gradient(gradient, rank, svd_niter=2):
    """Return a rank-limited ``U, S, V`` representation of a gradient."""
    gradient = gradient.detach().float()
    q = min(int(rank), gradient.shape[0], gradient.shape[1])
    if q <= 0:
        raise ValueError(f"gradient_rank must be positive, got {rank}")

    if q < min(gradient.shape):
        U, S, V = torch.svd_lowrank(gradient, q=q, niter=int(svd_niter))
        order = torch.argsort(S, descending=True)
        U, S, V = U[:, order], S[order], V[:, order]
    else:
        U, S, Vh = torch.linalg.svd(gradient, full_matrices=False)
        U, S, V = U[:, :q], S[:q], Vh[:q].T
    return U.cpu(), S.cpu(), V.cpu()


@torch.no_grad()
def _factorized_mean_svd(components, rank):
    """Compress the mean of low-rank matrices without densifying it."""
    if not components:
        raise ValueError("At least one low-rank component is required.")

    left_factors = []
    right_factors = []
    count = float(len(components))
    for U, S, V in components:
        left_factors.append(U.float() * (S.float() / count).unsqueeze(0))
        right_factors.append(V.float())
    L = torch.cat(left_factors, dim=1)
    R = torch.cat(right_factors, dim=1)

    Ql, Rl = torch.linalg.qr(L, mode="reduced")
    Qr, Rr = torch.linalg.qr(R, mode="reduced")
    core = Rl @ Rr.T
    Uc, S, Vhc = torch.linalg.svd(core, full_matrices=False)
    q = min(int(rank), S.numel())
    U = Ql @ Uc[:, :q]
    V = Qr @ Vhc[:q].T
    return U.contiguous(), S[:q].contiguous(), V.contiguous()


def _collect_dns_gradient(
    config,
    adapter_dir,
    gradient_rank,
    calibration_batches,
    calibration_batch_size,
    svd_niter,
    basis_eps,
    logger,
):
    """Collect and compress DNS gradients at the fixed W0 + W1 model."""
    # Fix calibration-data selection and torch.svd_lowrank randomness.
    set_global_seed(config.seed)
    tokenizer = _prepare_tokenizer(config)
    glue_data = _load_glue_data(config, tokenizer) if _is_glue(config) else None
    if _is_commonsense(config):
        calibration_sample_count = (
            int(calibration_batches) * int(calibration_batch_size)
        )
        train_data = _get_commonsense_calibration_data(
            config,
            tokenizer,
            total_samples=calibration_sample_count,
        )
        logger.info(
            "[ReGRA] random commonsense DNS calibration: "
            f"total={len(train_data)}, seed={config.seed}, "
            f"source={config.data_path}"
        )
    else:
        train_data, _ = _get_task_train_data(
            config, tokenizer, val_set_size=0, glue_data=glue_data
        )
    allowed_columns = set(tokenizer.model_input_names) | {"label", "labels"}
    removable = [
        name for name in train_data.column_names if name not in allowed_columns
    ]
    if removable:
        train_data = train_data.remove_columns(removable)

    data_loader = DataLoader(
        train_data,
        batch_size=int(calibration_batch_size),
        shuffle=False,
        collate_fn=_make_data_collator(config, tokenizer),
    )

    logger.info(
        "[ReGRA] Loading W0 + W1 for DNS calibration ..."
    )
    base_model = _load_task_model(
        config, tokenizer, glue_data, device_map="cuda"
    )
    model = PeftModel.from_pretrained(
        base_model, adapter_dir, autocast_adapter_dtype=False
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    stage1_state = _load_stage1_lora_state(adapter_dir)
    named_parameters = dict(model.named_parameters())
    components = {}
    energy_ratios = {}
    hooks = []
    target_parameters = []
    missing = []

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for a_name, A1 in stage1_state.items():
        if ".lora_A." not in a_name:
            continue
        b_name = a_name.replace(".lora_A.", ".lora_B.")
        base_name = a_name.replace(
            ".lora_A.default.weight", ".base_layer.weight"
        )
        if b_name not in stage1_state or base_name not in named_parameters:
            missing.append((a_name, base_name))
            continue

        left, right = _stage1_subspaces(
            A1, stage1_state[b_name], eps=basis_eps
        )
        parameter = named_parameters[base_name]
        parameter.requires_grad_(True)
        target_parameters.append(parameter)
        components[a_name] = []
        energy_ratios[a_name] = []

        def make_hook(key, left_basis, right_basis):
            def hook(param):
                if param.grad is None:
                    return
                dense_gradient = param.grad.detach().float()
                raw_norm_sq = dense_gradient.square().sum()
                projected = _project_dns_gradient(
                    dense_gradient, left_basis, right_basis
                )
                projected_norm_sq = projected.square().sum()
                ratio = projected_norm_sq / (raw_norm_sq + 1e-30)
                energy_ratios[key].append(float(ratio.cpu()))
                components[key].append(
                    _compress_gradient(projected, gradient_rank, svd_niter)
                )
                param.grad = None

            return hook

        if not hasattr(parameter, "register_post_accumulate_grad_hook"):
            raise RuntimeError(
                "This PyTorch version lacks register_post_accumulate_grad_hook; "
                "it is required for memory-safe residual calibration."
            )
        hooks.append(parameter.register_post_accumulate_grad_hook(
            make_hook(a_name, left, right)
        ))

    if missing or not components:
        raise ValueError(
            "Could not match Stage-1 LoRA tensors to base weights. "
            f"matched={len(components)}, missing_examples={missing[:3]}"
        )

    input_device = model.get_input_embeddings().weight.device
    completed_batches = 0
    try:
        for batch_index, batch in enumerate(data_loader):
            if batch_index >= int(calibration_batches):
                break
            model.zero_grad(set_to_none=True)
            batch = {
                key: value.to(input_device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            loss = model(**batch).loss
            loss.backward()
            completed_batches += 1
            logger.info(
                f"[ReGRA] DNS batch {completed_batches}/"
                f"{calibration_batches}, loss={loss.detach().float().item():.6f}"
            )
    finally:
        for hook in hooks:
            hook.remove()
        for parameter in target_parameters:
            parameter.grad = None
            parameter.requires_grad_(False)

    if completed_batches == 0:
        raise ValueError("No calibration batch was available for residual initialization.")
    incomplete = [
        name for name, values in components.items()
        if len(values) != completed_batches
    ]
    if incomplete:
        raise RuntimeError(
            "Some target weights did not receive all calibration gradients: "
            f"{incomplete[:3]}"
        )

    residual_svd = {
        name: _factorized_mean_svd(values, gradient_rank)
        for name, values in components.items()
    }
    ratios = [value for values in energy_ratios.values() for value in values]
    mean_ratio = sum(ratios) / len(ratios)
    logger.info(
        f"[ReGRA] DNS gradients: layers={len(residual_svd)}, "
        f"rank={gradient_rank}, batches={completed_batches}, "
        f"mean_retained_energy={mean_ratio:.6f}"
    )

    del model, base_model, named_parameters, components
    gc.collect()
    torch.cuda.empty_cache()
    return residual_svd, mean_ratio



def _stage1_subspaces(A1, B1, eps=1e-7):
    """Return exact nonzero left/right bases of ``delta_W1 = B1 @ A1``.

    The LoRA scaling ``alpha / rank`` is intentionally omitted because a
    nonzero scalar changes singular values but not singular subspaces.

    If ``B1 = Q_B R_B`` and ``A1.T = Q_A R_A``, then

    ``B1 @ A1 = Q_B @ (R_B @ R_A.T) @ Q_A.T``.

    Consequently, only the small core ``R_B @ R_A.T`` needs an SVD.  This is
    an exact low-rank factorization, not a randomized or truncated
    approximation.
    """
    A1 = A1.detach().float()
    B1 = B1.detach().float()
    if A1.ndim != 2 or B1.ndim != 2:
        raise ValueError(
            "Expected two LoRA matrices, got "
            f"A1.shape={tuple(A1.shape)}, B1.shape={tuple(B1.shape)}"
        )
    if B1.shape[1] != A1.shape[0]:
        raise ValueError(
            "Incompatible LoRA factors for B1 @ A1: "
            f"A1.shape={tuple(A1.shape)}, B1.shape={tuple(B1.shape)}"
        )

    output_dim = B1.shape[0]
    input_dim = A1.shape[1]
    if A1.numel() == 0 or B1.numel() == 0:
        return (
            B1.new_zeros((output_dim, 0)),
            A1.new_zeros((input_dim, 0)),
        )

    Q_B, R_B = torch.linalg.qr(B1, mode="reduced")
    Q_A, R_A = torch.linalg.qr(A1.T, mode="reduced")
    core = R_B @ R_A.T
    U_core, singular_values, Vh_core = torch.linalg.svd(
        core, full_matrices=False
    )
    if singular_values.numel() == 0 or singular_values[0] <= 0:
        return (
            B1.new_zeros((output_dim, 0)),
            A1.new_zeros((input_dim, 0)),
        )

    # Match the rank criterion previously applied to the dense update SVD.
    tolerance = max(output_dim, input_dim) * float(eps) * singular_values[0]
    effective_rank = int(
        torch.count_nonzero(singular_values > tolerance).item()
    )
    left = (Q_B @ U_core[:, :effective_rank]).contiguous()
    right = (Q_A @ Vh_core[:effective_rank].T).contiguous()
    return left, right


@torch.no_grad()
def _project_dns_gradient(gradient, left_basis, right_basis):
    """Return ``G - P_left G P_right`` using thin Stage-1 bases."""
    projected = gradient.detach().float().clone()
    if left_basis.shape[1] and right_basis.shape[1]:
        left = left_basis.to(projected.device, dtype=projected.dtype)
        right = right_basis.to(projected.device, dtype=projected.dtype)
        projected -= left @ (left.T @ projected @ right) @ right.T
    return projected


@torch.no_grad()
def _initialize_A_from_projected_gradient(model, residual_svd):
    """Initialize A2 with DNS right-singular directions and keep B2 zero.

    Singular values are deliberately ignored: every selected direction has
    unit weight, so the initialization encodes subspace directions only.
    """
    parameters = dict(model.named_parameters())
    initialized = 0
    missing = []
    for a_name, (_, _, V) in residual_svd.items():
        b_name = a_name.replace(".lora_A.", ".lora_B.")
        if a_name not in parameters or b_name not in parameters:
            missing.append((a_name, b_name))
            continue

        A_param = parameters[a_name]
        B_param = parameters[b_name]
        rank = A_param.shape[0]
        if (
            V.ndim != 2
            or V.shape[0] != A_param.shape[1]
            or V.shape[1] < rank
        ):
            raise ValueError(
                f"Projected-gradient SVD does not match {a_name}: "
                f"V={tuple(V.shape)}, A2={tuple(A_param.shape)}."
            )
        right_vectors = V[:, :rank].float().T
        if not torch.isfinite(right_vectors).all():
            raise ValueError(
                f"Non-finite DNS right-singular vectors for {a_name}."
            )
        A_param.copy_(right_vectors.to(
            device=A_param.device, dtype=A_param.dtype
        ))
        B_param.zero_()
        initialized += 1

    if missing or initialized == 0:
        raise ValueError(
            "Projected-gradient SVD does not match Stage-2 LoRA layers: "
            f"initialized={initialized}, missing_examples={missing[:3]}."
        )
    return initialized


class ReGRAStage2Trainer(transformers.Trainer):
    """A-only DNS initialization plus W1/W2 orthogonality."""

    def __init__(self, *args, stage1_state_dict, orth_lambda, orth_eps=1e-12,
                 residual_svd=None, projection_mode="dns",
                 orth_warmup_steps=0, basis_eps=1e-7, **kwargs):
        if projection_mode not in {"both", "dns"}:
            raise ValueError(f"ReGRA requires DNS, got {projection_mode!r}.")
        if not residual_svd:
            raise ValueError("ReGRA requires calibrated DNS gradients.")
        if orth_lambda < 0 or orth_warmup_steps < 0:
            raise ValueError("orth_lambda and warmup must be non-negative.")
        model = kwargs.get("model")
        if model is None:
            raise ValueError("ReGRAStage2Trainer requires model.")
        self.projected_a_layer_count = _initialize_A_from_projected_gradient(
            model, residual_svd
        )
        self.stage1_sd = stage1_state_dict
        self.orth_lambda, self.orth_eps = float(orth_lambda), float(orth_eps)
        self.orth_warmup_steps = int(orth_warmup_steps)
        self.basis_eps = float(basis_eps)
        super().__init__(*args, **kwargs)

        params = dict(self.model.named_parameters())
        self.lora_pairs, missing = [], []
        for a_name, A1 in self.stage1_sd.items():
            if ".lora_A." not in a_name:
                continue
            b_name = a_name.replace(".lora_A.", ".lora_B.")
            if b_name not in self.stage1_sd or a_name not in params or b_name not in params:
                missing.append((a_name, b_name))
                continue
            A2, B1, B2 = params[a_name], self.stage1_sd[b_name], params[b_name]
            if A1.shape[1] != A2.shape[1] or B1.shape[0] != B2.shape[0]:
                raise ValueError(f"Incompatible LoRA shapes for {a_name}.")
            self.lora_pairs.append((
                A1.to(device=A2.device, dtype=torch.float32),
                B1.to(device=B2.device, dtype=torch.float32),
                A2,
                B2,
            ))
        if missing or not self.lora_pairs:
            raise ValueError(
                f"LoRA layers do not match: matched={len(self.lora_pairs)}, "
                f"missing={missing[:3]}."
            )
        self._lm_loss_sum = self._orth_loss_sum = None
        self._weighted_orth_loss_sum = None
        self._loss_count = 0

    def _orthogonality_loss(self):
        """Return mean squared Frobenius cosine between W1 and W2."""
        losses = []
        for A1, B1, A2_param, B2_param in self.lora_pairs:
            A2, B2 = A2_param.float(), B2_param.float()
            cross_b = B1.T @ B2
            cross_a = A1 @ A2.T
            inner = torch.sum(cross_b * cross_a)
            norm1_sq = torch.sum((B1.T @ B1) * (A1 @ A1.T).T)
            norm2_sq = torch.sum((B2.T @ B2) * (A2 @ A2.T).T)
            losses.append(
                inner.square()
                / (norm1_sq * norm2_sq + self.orth_eps)
            )
        return torch.stack(losses).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )
        lm_loss, outputs = result if return_outputs else (result, None)
        orth_loss = self._orthogonality_loss()
        ramp = min(1.0, (float(self.state.global_step) + 1.0) /
                   self.orth_warmup_steps) if self.orth_warmup_steps else 1.0
        weighted_orth_loss = self.orth_lambda * ramp * orth_loss

        # Recent Transformers versions pass num_items_in_batch to causal-LM
        # models. In that path each micro-batch LM loss is already normalized
        # over the full accumulated batch, so Trainer sums (rather than divides)
        # the micro-batch losses. Scale the batch-independent regularizer once
        # per optimizer step and undo the partial-loss scaling only for logging.
        accumulation_steps = max(
            1, int(getattr(self, "current_gradient_accumulation_steps", 1))
        )
        trainer_sums_microbatch_losses = (
            model.training
            and kwargs.get("num_items_in_batch") is not None
            and self.model_accepts_loss_kwargs
        )
        regularizer_scale = (
            1.0 / accumulation_steps
            if trainer_sums_microbatch_losses
            else 1.0
        )
        lm_log_scale = (
            accumulation_steps if trainer_sums_microbatch_losses else 1.0
        )
        total_loss = lm_loss + regularizer_scale * weighted_orth_loss

        logged_lm_loss = lm_loss.detach().float() * lm_log_scale
        logged_orth_loss = orth_loss.detach().float()
        logged_weighted_orth_loss = weighted_orth_loss.detach().float()
        if self._loss_count == 0:
            self._lm_loss_sum = logged_lm_loss
            self._orth_loss_sum = logged_orth_loss
            self._weighted_orth_loss_sum = logged_weighted_orth_loss
        else:
            self._lm_loss_sum += logged_lm_loss
            self._orth_loss_sum += logged_orth_loss
            self._weighted_orth_loss_sum += logged_weighted_orth_loss
        self._loss_count += 1
        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs, *args, **kwargs):
        if self._loss_count:
            logs["lm_loss"] = (self._lm_loss_sum / self._loss_count).item()
            logs["orth_loss"] = (self._orth_loss_sum / self._loss_count).item()
            logs["weighted_orth_loss"] = (
                self._weighted_orth_loss_sum / self._loss_count
            ).item()
            self._lm_loss_sum = self._orth_loss_sum = None
            self._weighted_orth_loss_sum = None
            self._loss_count = 0
        return super().log(logs, *args, **kwargs)


def run_regra(config, config_dict):
    """Run DNS-projected A-only init plus ordinary Stage-2 training."""
    regra_config = getattr(config, "regra_config", None)
    if regra_config is None:
        raise ValueError("regra mode requires a 'regra_config' section.")

    stage = int(getattr(regra_config, "stage", 2))
    if stage not in (1, 2):
        raise ValueError(f"regra_config.stage must be 1 or 2, got {stage!r}")
    
    run_regra_stage1(config, config_dict)

    projection_mode = str(
        getattr(regra_config, "projection_mode", "dns")
    ).lower()
    if projection_mode not in {"both", "dns"}:
        raise ValueError(
            "regra supports projection_mode='dns' (or legacy alias 'both'), "
            f"got {projection_mode!r}."
        )

    basis_eps = float(getattr(regra_config, "basis_eps", 1e-7))
    if basis_eps <= 0:
        raise ValueError("regra_config.basis_eps must be positive.")

    rank = int(config.lora_config.rank)
    gradient_rank = int(getattr(regra_config, "gradient_rank", rank))
    gradient_batches = int(getattr(regra_config, "gradient_batches", 1))
    gradient_batch_size = int(getattr(regra_config, "gradient_batch_size", 1))
    gradient_svd_niter = int(getattr(regra_config, "gradient_svd_niter", 4))
    orth_lambda = float(getattr(regra_config, "orth_lambda", 0.1))
    orth_warmup_steps = int(getattr(regra_config, "orth_warmup_steps", 0))

    if gradient_rank < rank:
        raise ValueError(
            "regra_config.gradient_rank must be at least the LoRA rank: "
            f"gradient_rank={gradient_rank}, lora_rank={rank}."
        )
    if min(gradient_batches, gradient_batch_size) <= 0:
        raise ValueError(
            "gradient_batches and gradient_batch_size must be positive."
        )
    if gradient_svd_niter < 0:
        raise ValueError("gradient_svd_niter must be non-negative.")
    if orth_lambda < 0:
        raise ValueError("regra_config.orth_lambda must be non-negative.")
    if orth_warmup_steps < 0:
        raise ValueError("regra_config.orth_warmup_steps must be non-negative.")

    if isinstance(config_dict.get("regra_config"), dict):
        config_dict["regra_config"].setdefault("gradient_rank", gradient_rank)
        config_dict["regra_config"].setdefault("gradient_batches", gradient_batches)
        config_dict["regra_config"].setdefault(
            "gradient_batch_size", gradient_batch_size
        )
        config_dict["regra_config"].setdefault(
            "gradient_svd_niter", gradient_svd_niter
        )
        config_dict["regra_config"].setdefault(
            "orth_warmup_steps", orth_warmup_steps
        )

    logger = get_log(config.output_dir, "setting")
    stage1_adapter_dir = _find_adapter_dir(
        os.path.join(config.output_dir, "stage1")
    )
    logger.info("[ReGRA] Reusing the existing Stage-1 adapter.")
    logger.info(f"[ReGRA] stage1_adapter: {stage1_adapter_dir}")
    logger.info("[ReGRA] initialization projection: G_new = G - P_left G P_right")
    logger.info(f"[ReGRA] gradient_rank: {gradient_rank}")
    logger.info(f"[ReGRA] gradient_batches: {gradient_batches}")
    logger.info(f"[ReGRA] gradient_batch_size: {gradient_batch_size}")
    logger.info(
        "[ReGRA] initialization: "
        "A2 = V_dns.T, B2 = 0 (singular values ignored)"
    )
    logger.info(
        "[ReGRA] W1/W2 orthogonality loss: "
        "<W1,W2>_F^2 / (||W1||_F^2 ||W2||_F^2 + eps), "
        f"lambda={orth_lambda}, warmup_steps={orth_warmup_steps}"
    )
    logger.info("[ReGRA] training backward: ordinary LoRA gradients (no DNS)")
    logger.info("[ReGRA] model: W = W0 + W1 + W2")
    logger.info(
        "[ReGRA] DNS projectors: SVD(W1 = B1 A1); "
        "orthogonality: <W1, W2>_F = 0"
    )
    logger.info(f"[ReGRA] basis_eps: {basis_eps}")

    if not config.finetune:
        logger.info("[ReGRA] finetune=false; skip G_new calibration and training.")
        run_regra_eval(config)
        return

    residual_svd, retained_energy = _collect_dns_gradient(
        config=config,
        adapter_dir=stage1_adapter_dir,
        gradient_rank=gradient_rank,
        calibration_batches=gradient_batches,
        calibration_batch_size=gradient_batch_size,
        svd_niter=gradient_svd_niter,
        basis_eps=basis_eps,
        logger=logger,
    )
    logger.info(
        f"[ReGRA] G_new mean retained energy: {retained_energy:.6f}"
    )

    run_regra_stage2(config, config_dict, residual_svd)

    gc.collect()
    torch.cuda.empty_cache()
    run_regra_eval(config)


__all__ = [
    "ReGRAStage2Trainer",
    "run_regra",
    "_initialize_A_from_projected_gradient",
    "_project_dns_gradient",
]
