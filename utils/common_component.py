import os
import yaml

def mkdir(path: str):
    try:
        os.mkdir(path=path)
    except FileNotFoundError:
        os.system("mkdir -p " + path)

def yaml_write(yaml_dict: dict, yaml_file: str):
    root, _ = os.path.split(yaml_file)

    try:
        assert os.path.isdir(root)
    except AssertionError:
        mkdir(root)

    with open(yaml_file, "w", encoding="utf-8") as yaml_file:
        yaml.dump(yaml_dict, yaml_file)

def yaml_read(yaml_file: str):
    assert os.path.isfile(yaml_file)
    with open(yaml_file, "r") as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_dict
