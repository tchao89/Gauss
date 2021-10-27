"""-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab"""
import copy
import shelve

import yaml
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from gauss.data_clear.base_data_clear import BaseDataClear
from entity.dataset.base_dataset import BaseDataset

from utils.base import get_current_memory_gb
from utils.yaml_exec import yaml_read
from utils.reduce_data import reduce_data
from utils.Logger import logger
from utils.constant_values import ConstantValues


# 需要传入三个参数， 数模型的数据/非数模型的数据， yaml文件， base dataset
class PlainDataClear(BaseDataClear):
    def __init__(self, **params):
        """Construct a PlainDataClear.

        :param name: The name of the Component.
        :param strategy_dict: strategy for missing value. You can use 'mean', 'median', 'most_frequent' and 'constant',
        and if 'constant' is used, an efficient fill_value must be given.you can use two strategy_dict formats, for example:
        1 > {"model": {"name": "ftype"}, "category": {"name": 'most_frequent'}, "numerical": {"name": "mean"}, "bool": {"name": "most_frequent"}, "datetime": {"name": "most_frequent"}}
        2 > {"model": {"name": "feature"}, "feature 1": {"name": 'most_frequent'}, "feature 2": {"name": 'constant', "fill_value": 0}}
        But you can just use one of them, PlainDataClear object will use strict coding check programming.
        """

        super(PlainDataClear, self).__init__(name=params["name"], train_flag=params["train_flag"],
                                             enable=params["enable"], task_name=params["task_name"])

        self._feature_configure_path = params["feature_configure_path"]
        self._final_file_path = params["final_file_path"]
        # 序列化模型
        self._data_clear_configure_path = params["data_clear_configure_path"]

        self._strategy_dict = params["strategy_dict"]

        self._missing_values = np.nan

        self._default_cat_impute_model = SimpleImputer(missing_values=self._missing_values, strategy="most_frequent")
        self._default_num_impute_model = SimpleImputer(missing_values=self._missing_values, strategy="mean")

        self._impute_models = {}
        self._already_data_clear = None

    def _train_run(self, **entity):
        logger.info("Data clear component flag: " + str(self._enable))
        if self._enable is True:
            self._already_data_clear = True
            assert "train_dataset" in entity.keys()
            logger.info("Running clean() method and clearing, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            self._clean(dataset=entity["train_dataset"])

        else:
            self._already_data_clear = False

        logger.info("Data clearing feature configuration is generating, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        self.final_configure_generation()

        logger.info("Data clearing impute models serializing, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        self._data_clear_serialize()

    def _increment_run(self, **entity):
        assert ConstantValues.increment_dataset in entity.keys()
        dataset = entity[ConstantValues.increment_dataset]

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)

        feature_names = dataset.get_dataset().feature_names
        feature_conf = yaml_read(self._feature_configure_path)
        self._aberrant_modify(data=data)

        if self._enable is True:
            with shelve.open(self._data_clear_configure_path) as shelve_open:
                dc_model_list = shelve_open['impute_models']

            for col in feature_names:
                item_conf = feature_conf[col]
                if dc_model_list.get(col):
                    item_data = np.array(data[col]).reshape(-1, 1)

                    if "int" in item_conf['dtype']:
                        dc_model_list.get(col).fit(item_data.astype(np.int64))
                    elif "float" in item_conf['dtype']:
                        dc_model_list.get(col).fit(item_data.astype(np.float64))
                    else:
                        dc_model_list.get(col).fit(item_data)

                    item_data = dc_model_list.get(col).transform(item_data)
                    data[col] = item_data.reshape(1, -1).squeeze(axis=0)

    def _predict_run(self, **entity):
        assert "infer_dataset" in entity.keys()
        dataset = entity["infer_dataset"]

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)

        feature_names = dataset.get_dataset().feature_names
        feature_conf = yaml_read(self._feature_configure_path)
        self._aberrant_modify(data=data)

        if self._enable is True:
            with shelve.open(self._data_clear_configure_path) as shelve_open:
                dc_model_list = shelve_open['impute_models']

            for col in feature_names:
                item_conf = feature_conf[col]
                if dc_model_list.get(col):
                    item_data = np.array(data[col]).reshape(-1, 1)

                    if "int" in item_conf['dtype']:
                        dc_model_list.get(col).fit(item_data.astype(np.int64))
                    elif "float" in item_conf['dtype']:
                        dc_model_list.get(col).fit(item_data.astype(np.float64))
                    else:
                        dc_model_list.get(col).fit(item_data)

                    item_data = dc_model_list.get(col).transform(item_data)
                    data[col] = item_data.reshape(1, -1).squeeze(axis=0)

    def _clean(self, dataset: BaseDataset):
        data = dataset.get_dataset().data
        feature_names = dataset.get_dataset().feature_names

        assert isinstance(data, pd.DataFrame)
        self._aberrant_modify(data=data)

        feature_conf = yaml_read(self._feature_configure_path)

        for feature in feature_names:
            item_data = np.array(data[feature])
            # feature configuration, dict type
            item_conf = feature_conf[feature]

            if self._strategy_dict is not None:
                if self._strategy_dict["model"]["name"] == "ftype":
                    impute_model = SimpleImputer(missing_values=self._missing_values,
                                                 strategy=self._strategy_dict[item_conf['ftype']]["name"],
                                                 fill_value=self._strategy_dict[item_conf['ftype']].get("fill_value"),
                                                 add_indicator=True)

                else:
                    assert self._strategy_dict["model"]["name"] == "feature"
                    impute_model = SimpleImputer(missing_values=self._missing_values,
                                                 strategy=self._strategy_dict[feature]["name"],
                                                 fill_value=self._strategy_dict[feature].get("fill_value"),
                                                 add_indicator=True)

            else:

                if item_conf['ftype'] == "numerical":
                    impute_model = copy.deepcopy(self._default_num_impute_model)

                else:
                    assert item_conf['ftype'] in ["category", "bool", "datetime"]
                    impute_model = copy.deepcopy(self._default_cat_impute_model)

            item_data = item_data.reshape(-1, 1)

            if "int" in item_conf['dtype']:
                impute_model.fit(item_data.astype(np.int64))
            elif "float" in item_conf['dtype']:
                impute_model.fit(item_data.astype(np.float64))
            else:
                impute_model.fit(item_data)

            item_data = impute_model.transform(item_data)
            item_data = item_data.reshape(1, -1).squeeze(axis=0)

            self._impute_models[feature] = impute_model
            data[feature] = item_data
        reduce_data(dataframe=data)

    def _aberrant_modify(self, data: pd.DataFrame):
        feature_conf = yaml_read(self._feature_configure_path)

        for col in data.columns:
            dtype = feature_conf[col]["dtype"]
            check_nan = [self._type_check(item, dtype) for item in data[col]]

            if not all(check_nan):
                data[col] = data[col].where(check_nan)

    @classmethod
    def _type_check(cls, item, dtype):
        """
        this method is used to infer if a type of an object is int, float or string based on TypeInference object.
        :param item:
        :param dtype: dtype of a feature in feature configure file.
        :return:
        """
        assert dtype in ["int64", "float64", "string"]

        # When dtype is int, np.nan or string item can exist.
        if dtype == "int64":
            try:
                int(item)
                return True
            except ValueError:
                return False

        if dtype == "float64":
            try:
                float(item)
                return True
            except ValueError:
                return False
        return True

    def _data_clear_serialize(self):
        # 序列化label encoding模型字典
        with shelve.open(self._data_clear_configure_path) as shelve_open:
            shelve_open['impute_models'] = self._impute_models

    def final_configure_generation(self):
        feature_conf = yaml_read(yaml_file=self._feature_configure_path)

        with open(self._final_file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(feature_conf, yaml_file)

    @property
    def already_data_clear(self):
        return self._already_data_clear
