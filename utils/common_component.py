# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import re
import os
from typing import List, Any

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

def feature_list_generator(feature_conf):
    feature_dict = {}
    if isinstance(feature_conf, dict):
        feature_dict = feature_conf
    else:
        # name: FeatureItemConf
        feature_conf = feature_conf.feature_dict
        for item in feature_conf.keys():
            item = feature_conf[item]
            item_dict = {"name": item.name, "index": item.index, "dtype": item.dtype, "ftype": item.ftype, "used": item.used}
            feature_dict[item.name] = item_dict

    def get_item(dict_key):
        return feature_dict[dict_key]["index"]

    feature_list = []
    for item in feature_dict.keys():
        if feature_dict[item]["used"] is True:
            feature_list.append(item)
    feature_list.sort(key=get_item)

    return feature_list

def feature_list_selector(feature_conf: Any, feature_indexes: List[int]):
    feature_dict = {}
    if isinstance(feature_conf, dict):
        feature_dict = feature_conf
    else:
        # name: FeatureItemConf
        feature_conf = feature_conf.feature_dict
        for item in feature_conf.keys():
            item = feature_conf[item]
            item_dict = {"name": item.name, "index": item.index, "dtype": item.dtype, "ftype": item.ftype, "used": item.used}
            feature_dict[item.name] = item_dict

    def get_item(dict_key):
        return feature_dict[dict_key]["index"]

    used_features = []
    for item in feature_dict.keys():
        if feature_dict[item]["used"] is True:
            used_features.append(item)
        else:
            raise ValueError("Supervised selected features must have \" true\" used type.")
    used_features.sort(key=get_item)

    feature_list = []

    for index, item in enumerate(used_features):
        if index in feature_indexes:
            feature_list.append(item)
    feature_list.sort(key=get_item)

    return feature_list
