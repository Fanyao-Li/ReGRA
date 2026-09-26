import yaml
import argparse
from box import Box


def update_nested_dict(d, key_path, value):
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def is_valid_key(config, key_path):
    keys = key_path.split('.')
    d = config
    for key in keys:
        if key in d:
            d = d[key]
        else:
            return False
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Script to launch with a specific configuration.")
    parser.add_argument('-f', type=str, default=None, help='Path to the YAML configuration file.')
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args

def load_config(yaml_file=None):
    known_args, unknown_args = parse_args()
    
    if known_args.f is None:
        if yaml_file is not None:
            known_args.f = yaml_file
        else:
            raise ValueError("No yaml file given.")

    with open(known_args.f, 'r') as f:
        config = yaml.safe_load(f)

    # Now we can validate the unknown args against the config
    override_args = {}
    for arg in unknown_args:
        if arg.startswith('--'):
            print(arg[2:].split('='))
            key, value = arg[2:].split('=')
            if is_valid_key(config, key):
                override_args[key] = value
            else:
                # print(f"Warning: Ignoring invalid configuration key: {key}")
                assert False, f"Invalid configuration key: {key}"

    # Override configuration parameters if provided
    for key, value in override_args.items():
        # Convert value to appropriate type
        try:
            value = eval(value)
        except:
            pass
        update_nested_dict(config, key, value)
    
    # Print the updated configuration for debugging
    
    return config, Box(config)

def print_dict_paths(d, logger=None, parent_path=""):
    for key, value in d.items():
        current_path = f"{parent_path}.{key}" if parent_path else key
        if isinstance(value, dict):
            print_dict_paths(value, logger, current_path)
        else:
            if logger is None:
                print(f"{current_path}: {value}")
            else:
                logger.info(f"{current_path}: {value}")