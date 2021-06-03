# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab

from entity.entity import Entity
from utils.bunch import Bunch


class FeatureItemConf(object):

    def __init__(self, name: str = None, index: int = None, size=1, dtype="float32",
                 default_value=None, ftype="numerical"):
        assert dtype in ("int64", "float32", "string", "bool", "date")
        assert ftype in ("numerical", "category")

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


class FeatureConf(Entity):
    def __init__(self, name, file_path):
        # json file path
        self._file_path = file_path
        self._feature_dict = Bunch()
        super(FeatureConf, self).__init__(
            name=name,
        )

    def __repr__(self):
        pass

    def parse(self):
        pass

    def add_item_type(self, column_name: str, feature_item_conf: FeatureItemConf):
        self._feature_dict[column_name] = feature_item_conf

    @property
    def feature_dict(self):
        return self._feature_dict

    def reset_feature_type(self, key, ftype):
        assert (key in self._feature_dict)
        assert (ftype in ("numerical", "category"))
        pass
