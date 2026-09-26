import os

os.environ["WANDB_MODE"] = "disabled"

from utils.dev_parse import load_config


def main():
    config_dict, config = load_config("config/lora/math.yaml")
    mode = getattr(config, "mode", "lora")
    if mode == "lora":
        from mypeft.lora import run_lora

        run_lora(config, config_dict)
    elif mode == "regra":
        from mypeft.regra import run_regra

        run_regra(config, config_dict)
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Available implementations: lora, regra. "
            "Archived variant configs require their original implementations."
        )


if __name__ == "__main__":
    main()
