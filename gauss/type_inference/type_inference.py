# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
import re
import copy

import yaml

from utils.Logger import logger

import numpy as np
import pandas as pd

from entity.base_dataset import BaseDataset
from gauss.type_inference.base_type_inference import BaseTypeInference
from entity.feature_config import FeatureItemConf, FeatureConf


class TypeInference(BaseTypeInference):
    def __init__(self,
                 name: str,
                 task_name: str,
                 train_flag: bool,
                 # user
                 source_file_path="null",
                 # final
                 final_file_path: str = './final_configure.yaml',
                 final_file_prefix="final"):

        super(TypeInference, self).__init__(
            name=name,
            train_flag=train_flag,
            source_file_path=source_file_path,
            final_file_path=final_file_path,
            final_file_prefix=final_file_prefix
        )

        assert task_name in ["regression", "classification"]

        self.task_name = task_name
        self.ftype_list = []
        self.dtype_list = []

        self.epsilon = 0.00001
        self.dtype_threshold = 0.95
        self.categorical_threshold = 0.01

        if source_file_path != "null":
            self.init_feature_configure = FeatureConf(name="source feature path", file_path=source_file_path)
            self.init_feature_configure.parse()
        else:
            self.init_feature_configure = None

        self.final_feature_configure = FeatureConf(name='target feature path', file_path=final_file_path)

    def _train_run(self, **entity):
        assert "dataset" in entity.keys()

        self.dtype_inference(dataset=entity["dataset"])
        self.ftype_inference(dataset=entity["dataset"])

        self._check_init_final_conf()
        self.final_configure_generation()
        return self.final_feature_configure

    def _predict_run(self, **entity):
        pass

    def _bool_column_selector(self, feature_name: str, dataset: pd.DataFrame):

        if self.init_feature_configure is not None \
                and self.init_feature_configure.feature_dict[feature_name].ftype == 'bool' \
                and self.init_feature_configure.feature_dict[feature_name].dtype == 'int64' or 'string':

            column_unique = list(set(dataset[feature_name]))

            if len(column_unique) == 2:
                self.final_feature_configure.feature_dict[feature_name].ftype = 'bool'

    def _datetime_column_selector(self, feature_name: str, dataset: pd.DataFrame):

        def datetime_map(x):
            x = str(x)

            if re.search(r"(\d{4}-\d{1,2}-\d{1,2})", x):
                return True
            else:
                return False

        if self.init_feature_configure is not None \
                and self.init_feature_configure.feature_dict[feature_name].ftype == 'datetime' \
                and self.init_feature_configure.feature_dict[feature_name].dtype == 'string':

            column_unique = list(set(dataset[feature_name]))
            print(map(datetime_map, column_unique))
            if all(map(datetime_map, column_unique)):
                self.final_feature_configure.feature_dict[feature_name].ftype = 'datetime'

    def ftype_inference(self, dataset: BaseDataset):
        data = dataset.get_dataset().data

        for index, column in enumerate(data.columns):

            if self.final_feature_configure.feature_dict[column].dtype == 'string':
                self.final_feature_configure.feature_dict[column].ftype = 'category'

            elif self.final_feature_configure.feature_dict[column].dtype == 'float64':
                self.final_feature_configure.feature_dict[column].ftype = 'numerical'
            else:
                assert self.final_feature_configure.feature_dict[column].dtype == 'int64'

                categorical_detect = len(pd.unique(data[column]))/data[column].shape[0]

                if categorical_detect < self.categorical_threshold:
                    self.final_feature_configure.feature_dict[column].ftype = 'category'
                else:
                    self.final_feature_configure.feature_dict[column].ftype = 'numerical'

            self._bool_column_selector(feature_name=column, dataset=data)
            self._datetime_column_selector(feature_name=column, dataset=data)

        return self.final_feature_configure

    def dtype_inference(self, dataset: BaseDataset):

        data = dataset.get_dataset().data
        target = dataset.get_dataset().target

        assert isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)
        assert isinstance(target, pd.DataFrame) or isinstance(target, pd.Series)

        data_dtypes = data.dtypes

        for col_index, column in enumerate(data):

            feature_item_configure = FeatureItemConf()
            feature_item_configure.name = column
            feature_item_configure.index = col_index

            if data_dtypes[col_index] == "int64":
                feature_item_configure.dtype = "int64"

            elif data_dtypes[col_index] == "float64":
                int_count = 0

                for item in data[column]:
                    if not np.isnan(item) and abs(item - int(item)) < self.epsilon:
                        int_count += 1

                if int_count + data[column].isna().sum() == data[column].shape[0]:
                    feature_item_configure.dtype = "int64"

                else:
                    feature_item_configure.dtype = "float64"

            elif data_dtypes[col_index] == 'object':

                str_count = 0
                int_count = 0
                float_count = 0
                str_coordinate = []
                float_coordinate = []

                for index, item in enumerate(data[column]):

                    try:
                        float(item)
                        float_count += 1
                        float_coordinate.append(index)
                    except ValueError:
                        str_count += 1
                        str_coordinate.append(index)

                if float_count/data.shape[0] > self.dtype_threshold:

                    feature_item_configure.dtype = 'float64'
                    detect_column = copy.deepcopy(data[column])
                    detect_column.loc[str_coordinate] = detect_column.iloc[str_coordinate].apply(lambda x: np.nan)
                    detect_column = pd.to_numeric(detect_column)

                    for item in detect_column:
                        if not np.isnan(item) and abs(item - int(item)) < self.epsilon:
                            int_count += 1
                    if int_count + detect_column.isna().sum() == detect_column.shape[0]:
                        feature_item_configure.dtype = "int64"
                else:
                    feature_item_configure.dtype = 'string'

            self.final_feature_configure.add_item_type(column_name=column, feature_item_conf=feature_item_configure)

        for label_index, label in enumerate(target):
            feature_item_configure = FeatureItemConf()

            if self.task_name == 'regression':
                assert target[label].dtypes == 'float64' and target[label].isna().sum() == 0

            if self.task_name == 'classification':
                assert target[label].dtypes == 'int64'

            feature_item_configure.name = label
            feature_item_configure.index = label_index
            feature_item_configure.dtype = str(target[label].dtype)
            self.final_feature_configure.add_item_type(column_name=label, feature_item_conf=feature_item_configure)

        return self.final_feature_configure

    def _check_init_final_conf(self):
        assert self.init_feature_configure is not None

        for item in self.init_feature_configure.feature_dict.items():
            if self.final_feature_configure.feature_dict.get(item[0]):
                exception = False

                if item[1].name and item[1].name != self.final_feature_configure.feature_dict[item[0]].name:
                    logger.info(item[0] + " feature's name is different between yaml file and type inference.")
                    exception = True

                if item[1].index and item[1].index != self.final_feature_configure.feature_dict[item[0]].index:
                    logger.info(item[0] + " feature's index is different between yaml file and type inference.")
                    exception = True

                if item[1].dtype and item[1].dtype != self.final_feature_configure.feature_dict[item[0]].dtype:
                    logger.info(item[0] + " feature's dtype is different between yaml file and type inference.")
                    exception = True

                if item[1].ftype and item[1].ftype != self.final_feature_configure.feature_dict[item[0]].ftype:
                    logger.info(item[0] + " feature's ftype is different between yaml file and type inference.")
                    exception = True

                if not exception:
                    logger.info('Customized feature ' + item[0] + " matches type inference. ")
            else:
                logger.info(item[0] + "feature dose not exist in type inference.")

    def final_configure_generation(self):
        yaml_dict = {}

        for item in self.final_feature_configure.feature_dict.items():
            item_dict = {"name": item[1].name, "index": item[1].index, "dtype": item[1].dtype, "ftype": item[1].ftype, "size": item[1].size}
            yaml_dict[item[0]] = item_dict

        with open(self._final_file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(yaml_dict, yaml_file)
