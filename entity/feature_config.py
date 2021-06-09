# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
import os
import yaml

from entity.entity import Entity
from utils.bunch import Bunch


class FeatureItemConf(object):
    def __init__(self, name: str = None, index: int = None, size=1, dtype=None,
                 default_value=None, ftype=None):
        assert dtype in (None, "int64", "float64", "string")
        assert ftype in (None, "numerical", "category", "bool")

        self._name = name
        self._dtype = dtype
        self._index = index
        self._size = size
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

class FeatureConf(Entity):
    def __init__(self, name, file_path):
        super(FeatureConf, self).__init__(
            name=name,
        )

        # yaml file path
        self._file_path = file_path

        self._feature_dict = Bunch()

    def __repr__(self):
        if self._file_path is not None:
            self.parse()
            return str(self._feature_dict)
        else:
            return None

    def parse(self):
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

        for item in init_conf['transforms']['categorical_features']:
            if self._feature_dict.get(item):
                self._feature_dict[item].ftype = "category"
            else:
                item_configure = FeatureItemConf()
                item_configure.ftype = "category"
                self._feature_dict[item] = item_configure

        for item in init_conf['transforms']['numerical_features']:
            if self._feature_dict.get(item):
                self._feature_dict[item].ftype = "numerical"
            else:
                item_configure = FeatureItemConf()
                item_configure.ftype = "numerical"
                self._feature_dict[item] = item_configure

        for item in init_conf['transforms']['bool_features']:
            if self._feature_dict.get(item):
                self._feature_dict[item].ftype = "bool"
            else:
                item_configure = FeatureItemConf()
                item_configure.ftype = "bool"
                self._feature_dict[item] = item_configure

        for item in init_conf['transforms']['datetime_features']:
            if self._feature_dict.get(item):
                self._feature_dict[item].ftype = "datetime"
            else:
                item_configure = FeatureItemConf()
                item_configure.ftype = "datetime"
                self._feature_dict[item] = item_configure
        print(self._feature_dict['time'].name)
        print(self._feature_dict['time'].dtype)
        print(self._feature_dict['time'].ftype)
        return self

    def add_item_type(self, column_name: str, feature_item_conf: FeatureItemConf):
        self._feature_dict[column_name] = feature_item_conf

    @property
    def feature_dict(self):
        return self._feature_dict

    def reset_feature_type(self, key, ftype):
        assert (key in self._feature_dict)
        assert (ftype in ("numerical", "category", "bool"))
