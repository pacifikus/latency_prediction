import yaml


def read_config(cfg_path: str) -> dict:
    with open(cfg_path, 'r') as f_open:
        return yaml.load(f_open, Loader=yaml.SafeLoader)
