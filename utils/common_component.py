import os
import yaml

def mkdir(path: str):
    os.mkdir(path=path)

def yaml_write(yaml_dict: dict, yaml_file: str):
    with open(yaml_file, "w", encoding="utf-8") as yaml_file:
        yaml.dump(yaml_dict, yaml_file)

def yaml_read(yaml_file: str):
    assert os.path.isfile(yaml_file)
    with open(yaml_file, "r") as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_dict
