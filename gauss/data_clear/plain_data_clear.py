# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import os

import yaml
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from gauss.data_clear.base_data_clear import BaseDataClear
from entity.base_dataset import BaseDataset


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
        assert os.path.isfile(self._feature_configure_path)

        self._strategy_dict = params["strategy_dict"]

        self.missing_values = np.nan

        self.default_cat_impute_model = SimpleImputer(missing_values=self.missing_values, strategy="most_frequent")
        self.default_num_impute_model = SimpleImputer(missing_values=self.missing_values, strategy="mean")

    def set_name(self, name):
        self._name = name

    def set_train_flag(self, train_flag: bool):
        self._train_flag = train_flag

    def set_enable(self, enable: bool):
        self._enable = enable

    def set_model_name(self, model_name):
        self._model_name = model_name

    def set_feature_configure_path(self, configure_path):
        self._feature_configure_path = configure_path

    def set_strategy_dict(self, strategy_dict):
        self._strategy_dict = strategy_dict

    def _train_run(self, **entity):
        if self._model_name == "tree_model":
            assert "dataset" in entity.keys()
            self._clean(dataset=entity["dataset"])

    def _predict_run(self, **entity):
        if self._model_name == "tree_model":
            assert "dataset" in entity.keys()
            self._clean(dataset=entity["dataset"])

    def _clean(self, dataset: BaseDataset):
        data = dataset.get_dataset().data
        feature_names = dataset.get_dataset().feature_names

        assert isinstance(data, pd.DataFrame)

        feature_conf_file = open(self._feature_configure_path, 'r', encoding='utf-8')
        feature_conf = feature_conf_file.read()
        feature_conf = yaml.load(feature_conf, Loader=yaml.FullLoader)

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

            data[feature] = item_data

    def configure_generation(self):
        """
        This method will generate data clear configuration from machine learning model name.
        :return:
        """
        pass
