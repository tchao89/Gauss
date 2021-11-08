"""-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab"""
from __future__ import annotations

import os
import yaml

from entity.entity import Entity
from utils.bunch import Bunch
from utils.yaml_exec import yaml_read
from utils.yaml_exec import yaml_write


class FeatureItemConf(object):
    def __init__(self, name: str = None, index: int = None, size=1, dtype=None,
                 default_value=None, ftype=None, used=None):

        assert dtype in (None, "int64", "float64", "string")
        assert ftype in (None, "numerical", "category", "bool", "datetime")

        self._name = name
        self._dtype = dtype
        self._index = index
        self._size = size
        if used is None:
            self._used = True
        else:
            self._used = used

        if default_value is None:
            if dtype == "string" or dtype == "date":
                self.default_value = "UNK"
            else:
                self.default_value = 0
        else:
            self.default_value = default_value

        self._ftype = ftype

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: str):
        self._dtype = dtype

    @property
    def ftype(self):
        return self._ftype

    @ftype.setter
    def ftype(self, ftype: str):
        self._ftype = ftype

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size

    @property
    def used(self):
        return self._used

    @used.setter
    def used(self, used: bool):
        self._used = used

class FeatureConf(Entity):
    def __init__(self, **params):
        super(FeatureConf, self).__init__(
            name=params["name"],
        )

        # yaml file path
        self._file_path = params.get("file_path")
        self._feature_dict = Bunch()

    def __repr__(self):
        pass

    def parse(self, method=None):
        assert method in ["user", "system"]
        if method == "user":
            assert os.path.isfile(self._file_path)

            init_conf_file = open(self._file_path, 'r', encoding='utf-8')
            init_conf = init_conf_file.read()
            init_conf = yaml.load(init_conf, Loader=yaml.FullLoader)
            init_conf_file.close()

            for item in init_conf['features'].items():
                if item[1]['size'] != 1:
                    raise ValueError("Size of each feature must be 1.")
                else:
                    item_configure = FeatureItemConf(name=item[0], dtype=item[1]['dtype'], index=item[1]['index'], size=item[1]['size'])
                    self._feature_dict[item[0]] = item_configure

            if len(list(set(init_conf['transforms']['categorical_features']))) != len(init_conf['transforms']['categorical_features'])\
                    or len(list(set(init_conf['transforms']['numerical_features']))) != len(init_conf['transforms']['numerical_features'])\
                    or len(list(set(init_conf['transforms']['bool_features']))) != len(init_conf['transforms']['bool_features']):

                raise ValueError("Duplicate keys in transformers.")

            if init_conf['transforms']['categorical_features'] is not None:
                for item in init_conf['transforms']['categorical_features']:
                    if self._feature_dict.get(item):
                        self._feature_dict[item].ftype = "category"
                    else:
                        item_configure = FeatureItemConf()
                        item_configure.ftype = "category"
                        self._feature_dict[item] = item_configure

            if init_conf['transforms']['numerical_features'] is not None:
                for item in init_conf['transforms']['numerical_features']:
                    if self._feature_dict.get(item):
                        self._feature_dict[item].ftype = "numerical"
                    else:
                        item_configure = FeatureItemConf()
                        item_configure.ftype = "numerical"
                        self._feature_dict[item] = item_configure

            if init_conf['transforms']['bool_features'] is not None:
                for item in init_conf['transforms']['bool_features']:
                    if self._feature_dict.get(item):
                        self._feature_dict[item].ftype = "bool"
                    else:
                        item_configure = FeatureItemConf()
                        item_configure.ftype = "bool"
                        self._feature_dict[item] = item_configure

            if init_conf['transforms']['datetime_features'] is not None:
                for item in init_conf['transforms']['datetime_features']:
                    if self._feature_dict.get(item):
                        self._feature_dict[item].ftype = "datetime"
                    else:
                        item_configure = FeatureItemConf()
                        item_configure.ftype = "datetime"
                        self._feature_dict[item] = item_configure
        else:
            self._feature_dict = Bunch()
            feature_dict = yaml_read(yaml_file=self._file_path)

            for key in feature_dict.keys():
                feature_item = feature_dict[key]
                item_configure = FeatureItemConf(name=feature_item["name"],
                                                 dtype=feature_item["dtype"],
                                                 ftype=feature_item["ftype"],
                                                 index=feature_item["index"],
                                                 used=feature_item["used"])

                self._feature_dict[key] = item_configure
            return self

    def add_item_type(self, column_name: str, feature_item_conf: FeatureItemConf):
        self._feature_dict[column_name] = feature_item_conf

    @property
    def feature_dict(self):
        return self._feature_dict

    # check function
    def reset_feature_type(self, key, ftype):
        assert (key in self._feature_dict)
        assert (ftype in ("numerical", "category", "bool"))

    def write(self, save_path="./feature_conf.yaml"):
        assert self._feature_dict is not None
        features = {}

        feature_dict = self._feature_dict
        for item_conf in feature_dict.keys():
            item_dict = {"name": item_conf.name, "dtype": item_conf.dtype, "ftype": item_conf.ftype,
                         "index": item_conf.index, "used": item_conf.used}
            features[item_conf.name] = item_dict
        yaml_write(yaml_dict=features, yaml_file=save_path)

    def feature_select(self, feature_list=None, use_index_flag=None):
        if feature_list is None:
            for feature in self._feature_dict.keys():
                self._feature_dict[feature].used = True
            return

        assert isinstance(use_index_flag, bool)
        if use_index_flag is True:
            for feature in self._feature_dict.keys():
                if self._feature_dict[feature].index not in feature_list:
                    self._feature_dict[feature].used = False
                else:
                    assert self._feature_dict[feature].used is True
        else:
            for feature in self._feature_dict.keys():
                if self._feature_dict[feature].name not in feature_list:
                    self._feature_dict[feature].used = False
                else:
                    assert self._feature_dict[feature].used is True

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        assert os.path.isfile(file_path)
        self._file_path = file_path
