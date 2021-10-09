"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
from typing import Any


def generate_feature_index(feature_conf: Any):
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
            feature_list.append(get_item(item))
    feature_list.sort()
    return feature_list
