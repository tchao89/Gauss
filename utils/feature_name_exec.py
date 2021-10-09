"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
from typing import List, Any

def generate_feature_list(feature_conf):
    feature_dict = {}
    if isinstance(feature_conf, dict):
        feature_dict = feature_conf
    else:
        # name: FeatureItemConf
        feature_conf = feature_conf.feature_dict
        for item in feature_conf.keys():
            item = feature_conf[item]
            item_dict = {"name": item.name,
                         "index": item.index,
                         "dtype": item.dtype,
                         "ftype": item.ftype,
                         "used": item.used}
            feature_dict[item.name] = item_dict

    def get_item(dict_key):
        return feature_dict[dict_key]["index"]

    feature_list = []
    for item in feature_dict.keys():
        if feature_dict[item]["used"] is True:
            feature_list.append(item)
    feature_list.sort(key=get_item)

    return feature_list

def select_feature_list(feature_conf: Any, feature_indexes: List[int]):
    feature_dict = {}
    if isinstance(feature_conf, dict):
        feature_dict = feature_conf
    else:
        # name: FeatureItemConf
        feature_conf = feature_conf.feature_dict
        for item in feature_conf.keys():
            item = feature_conf[item]
            item_dict = {"name": item.name,
                         "index": item.index,
                         "dtype": item.dtype,
                         "ftype": item.ftype,
                         "used": item.used}
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

def generate_categorical_list(feature_conf: Any):
    feature_dict = {}
    if isinstance(feature_conf, dict):
        feature_dict = feature_conf
    else:
        # name: FeatureItemConf
        feature_conf = feature_conf.feature_dict
        for item in feature_conf.keys():
            item = feature_conf[item]
            item_dict = {"name": item.name,
                         "index": item.index,
                         "dtype": item.dtype,
                         "ftype": item.ftype,
                         "used": item.used}
            feature_dict[item.name] = item_dict

    def get_item(dict_key):
        return feature_dict[dict_key]["index"]

    feature_list = []
    for item in feature_dict.keys():
        if feature_dict[item]["used"] is True:
            if feature_dict[item]["ftype"] == "category" or feature_dict[item]["ftype"] == "bool":
                feature_list.append(item)
    feature_list.sort(key=get_item)
    return feature_list
