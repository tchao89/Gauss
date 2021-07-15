# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import shelve

import yaml
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from gauss.data_clear.base_data_clear import BaseDataClear
from entity.dataset.base_dataset import BaseDataset

from utils.common_component import yaml_read


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

        super(PlainDataClear, self).__init__(name=params["name"], train_flag=params["train_flag"], enable=params["enable"])

        self._model_name = params["model_name"]
        self._feature_configure_path = params["feature_configure_path"]
        self._final_file_path = params["final_file_path"]
        # 序列化模型
        self._data_clear_configure_path = params["data_clear_configure_path"]

        self._strategy_dict = params["strategy_dict"]

        self.missing_values = np.nan

        self.default_cat_impute_model = SimpleImputer(missing_values=self.missing_values, strategy="most_frequent")
        self.default_num_impute_model = SimpleImputer(missing_values=self.missing_values, strategy="mean")

        self.impute_models = {}

    def _train_run(self, **entity):
        if self._enable:
            assert "dataset" in entity.keys()
            self._clean(dataset=entity["dataset"])
        self.final_configure_generation()
        self._data_clear_serialize()

    def _predict_run(self, **entity):
        data_clear_conf = yaml_read(self._feature_configure_path)
        assert "plain_data_clear" in data_clear_conf.keys()

        if data_clear_conf["plain_data_clear"] is True:
            assert "dataset" in entity.keys()
            dataset = entity["dataset"]

            data = dataset.get_dataset().data
            assert isinstance(data, pd.DataFrame)

            feature_names = dataset.get_dataset().feature_names
            with shelve.open(self._data_clear_configure_path) as shelve_open:
                dc_model_list = shelve_open['impute_models']

            for col in feature_names:
                if dc_model_list.get(col):
                    data[col] = dc_model_list.get(col).transform(data[col])

    def _clean(self, dataset: BaseDataset):
        data = dataset.get_dataset().data
        feature_names = dataset.get_dataset().feature_names

        assert isinstance(data, pd.DataFrame)

        feature_conf = yaml_read(self._feature_configure_path)

        for feature in feature_names:
            item_data = data[feature].values
            # feature configuration, dict type
            item_conf = feature_conf[feature]

            if self._strategy_dict is not None:
                if self._strategy_dict["model"]["name"] == "ftype":
                    impute_model = SimpleImputer(missing_values=self.missing_values,
                                                 strategy=self._strategy_dict[item_conf['ftype']]["name"],
                                                 fill_value=self._strategy_dict[item_conf['ftype']].get("fill_value"),
                                                 add_indicator=True)

                else:
                    assert self._strategy_dict["model"]["name"] == "feature"
                    impute_model = SimpleImputer(missing_values=self.missing_values,
                                                 strategy=self._strategy_dict[feature]["name"],
                                                 fill_value=self._strategy_dict[feature].get("fill_value"),
                                                 add_indicator=True)

            else:

                if item_conf['ftype'] == "numerical":
                    impute_model = self.default_num_impute_model

                else:
                    assert item_conf['ftype'] in ["category", "bool", "datetime"]
                    impute_model = self.default_cat_impute_model

            item_data = item_data.reshape(-1, 1)
            impute_model = impute_model.fit(item_data)

            item_data = impute_model.transform(item_data)
            item_data = item_data.reshape(1, -1).squeeze(axis=0)

            self.impute_models[feature] = impute_model

            data[feature] = item_data

    def _data_clear_serialize(self):
        # 序列化label encoding模型字典
        with shelve.open(self._data_clear_configure_path) as shelve_open:
            shelve_open['impute_models'] = self.impute_models

    def final_configure_generation(self):
        feature_conf = yaml_read(yaml_file=self._feature_configure_path)

        assert isinstance(self._enable, bool)
        feature_conf["plain_data_clear"] = self._enable

        with open(self._final_file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(feature_conf, yaml_file)
