# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import os
from typing import List

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

def feature_list_generator(feature_dict: dict):
    def get_item(dict_key):
        return feature_dict[dict_key]["index"]

    feature_list = []
    for item in feature_dict.keys():
        if feature_dict[item]["used"] is True:
            feature_list.append(item)
    feature_list.sort(key=get_item)

    return feature_list

def feature_list_selector(feature_dict: dict, feature_indexes: List[int]):
    def get_item(dict_key):
        return feature_dict[dict_key]["index"]

    used_features = []
    for item in feature_dict.keys():
        if feature_dict[item]["used"] is True:
            used_features.append(item)
    used_features.sort(key=get_item)

    feature_list = []

    for index, item in enumerate(used_features):
        if index in feature_indexes:
            feature_list.append(item)
    feature_list.sort(key=get_item)

    return feature_list
