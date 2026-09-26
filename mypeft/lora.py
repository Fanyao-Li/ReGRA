import os
import yaml
import torch
import socket
import transformers
from datetime import datetime
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from utils.logger import get_log
from utils.dev_parse import print_dict_paths
from utils.finetune_utils import get_data
from utils.evaluate_utils import MATH_DATASETS, COMMONSENSE_DATASETS, evaluate_dataset
from utils.train_utils import set_global_seed, convert_target_modules, get_trainable_params_numbers, print_delta_time, convert_lora_params_dtype


def run_lora(config, config_dict):
    """Standard LoRA fine-tuning pipeline."""
    logger = get_log(config.output_dir, "setting")
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)

    ## prepare
    server_name = socket.gethostname()
    logger.info(f"server_name: {server_name}")
    print_dict_paths(config_dict, logger)
    train_config = config.train_config
    test_config = config.test_config
    lora_config = config.lora_config
    is_glue = str(config.datamode).lower() == "glue"
    if is_glue:
        target_modules = lora_config.target_modules
        if isinstance(target_modules, str):
            target_modules = [
                name.strip() for name in target_modules.split(",") if name.strip()
            ]
        lora_config.target_modules = target_modules
    else:
        lora_config.target_modules = convert_target_modules(
            lora_config.target_modules
        )
    set_global_seed(config.seed)
    ft_start_time = datetime.now()

    ## model
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    glue_data = None
    if is_glue:
        from utils.glue_utils import load_glue_data

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("GLUE sequence classification requires a pad or EOS token.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        glue_data = load_glue_data(
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
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model,
            num_labels=glue_data["num_labels"],
            problem_type=(
                "regression" if glue_data["is_regression"]
                else "single_label_classification"
            ),
            # Match the standard GLUE/OPLoRA precision setup: keep model
            # parameters in FP32 and let Trainer(bf16=True) autocast compute.
            # This also initializes a missing classification head safely.
            dtype=torch.float32,
            device_map="auto",
            ignore_mismatched_sizes=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

    ## lora
    peft_kwargs = dict(
        r=lora_config.rank,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        use_dora=getattr(lora_config, "use_dora", False),
    )
    if is_glue:
        peft_kwargs.update(
            task_type=TaskType.SEQ_CLS,
            modules_to_save=["classifier", "score"],
        )
    peft_config = LoraConfig(**peft_kwargs)
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    convert_lora_params_dtype(model, dtype=lora_config.dtype)
    logger.info(model)
    rate = get_trainable_params_numbers(model, path=os.path.join(config.output_dir, "num_params.json"))
    logger.info(rate)

    ## print trainable names
    logger.info("### Trainable Parameters:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad is True:
            logger.info(f"{name}: {parameter.dtype}")

    ## finetune
    if config.finetune:
        if is_glue:
            train_data = glue_data["train_dataset"]
            val_data = glue_data["eval_dataset"]
        elif str(config.datamode).lower() == "code":
            from utils.code_utils import load_codefeedback_train

            train_data, val_data = load_codefeedback_train(
                data_path=config.data_path,
                tokenizer=tokenizer,
                max_length=train_config.max_length,
                train_on_inputs=train_config.train_on_inputs,
                val_set_size=train_config.val_set_size,
                seed=config.seed,
                max_train_samples=getattr(
                    train_config, "code_max_train_samples", 100000
                ),
            )
        else:
            train_data, val_data = get_data(
                config.data_path,
                train_config.val_set_size,
                tokenizer,
                train_config.max_length,
                train_config.train_on_inputs
            )
        has_validation = is_glue or train_config.val_set_size > 0
        best_metric = None
        if is_glue:
            glue_task = glue_data["task_name"]
            if glue_task == "stsb":
                best_metric = "combined_score"
            elif glue_task == "cola":
                best_metric = "matthews_correlation"
            elif glue_task in {"mrpc", "qqp"}:
                best_metric = "combined_score"
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
            output_dir=os.path.join(config.output_dir, "finetuned_result"),
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
            data_collator = transformers.DataCollatorWithPadding(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
        else:
            data_collator = transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            )
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            data_collator=data_collator,
            **trainer_kwargs,
        )
        trainer.train()
        model.save_pretrained(os.path.join(config.output_dir, "finetuned_result"))
        print_delta_time(ft_start_time, logger)

    ## evaluate
    if config.evaluate:
        model.eval()
        if test_config.merge:  # some methods do not support merge
            model = model.merge_and_unload()
        torch.cuda.empty_cache()
        test_data = str(config.datamode).lower()
        set_global_seed(config.seed)
        eval_start_time = datetime.now()

        if test_data == "glue":
            from utils.glue_utils import (
                evaluate_glue_trainer,
                make_glue_compute_metrics,
            )

            if glue_data is None:
                raise RuntimeError("GLUE data was not initialized.")
            evaluator = transformers.Trainer(
                model=model,
                args=transformers.TrainingArguments(
                    output_dir=os.path.join(config.output_dir, "glue_eval_tmp"),
                    per_device_eval_batch_size=int(test_config.test_batch_size),
                    bf16=True,
                    report_to=[],
                ),
                data_collator=transformers.DataCollatorWithPadding(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt"
                ),
                compute_metrics=make_glue_compute_metrics(
                    glue_data["task_name"], glue_data["is_regression"]
                ),
            )
            evaluate_glue_trainer(
                evaluator,
                glue_data,
                os.path.join(config.output_dir, "evaluated_result"),
                logger,
            )
            print_delta_time(eval_start_time, logger)
            return

        if test_data == "code":
            from utils.code_utils import evaluate_mbpp_model

            summary = evaluate_mbpp_model(
                model=model,
                tokenizer=tokenizer,
                output_dir=os.path.join(config.output_dir, "evaluated_result", "mbpp"),
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
            print_delta_time(eval_start_time, logger)
            return

        if test_data == "math":
            test_datasets = MATH_DATASETS
        else:
            test_datasets = COMMONSENSE_DATASETS

        if test_config.test_dataset_ids != "":
            test_datasets = [
                test_datasets[int(i)] for i in test_config.test_dataset_ids
            ]

        for test_dataset in test_datasets:
            accuracy = evaluate_dataset(
                model, tokenizer, test_data, test_dataset,
                test_config.test_batch_size,
                os.path.join(config.output_dir, "evaluated_result"),
            )
            if accuracy < 0.01:
                logger.info("stop evaluate due to poor accuracy.")
                break
        print_delta_time(eval_start_time, logger)
