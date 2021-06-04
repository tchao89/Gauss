# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
import copy

import numpy as np
import pandas as pd

from utils.bunch import Bunch
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
                 target_file_path: str = './',
                 target_file_prefix="target"):

        super(TypeInference, self).__init__(
            name=name,
            train_flag=train_flag,
            source_file_path=source_file_path,
            target_file_path=target_file_path,
            target_file_prefix=target_file_prefix
        )

        assert task_name in ["regression", "classification"]

        self.task_name = task_name
        self.ftype_list = []
        self.dtype_list = []

        self.epsilon = 0.00001
        self.dtype_threshold = 0.95
        self.categorical_threshold = 0.01

        if source_file_path != "null":
            self.feature_configure = FeatureConf(name="source feature path", file_path=source_file_path)
        else:
            self.feature_configure = FeatureConf(name="Feature Configure", file_path="./")

        self.target_feature_configure = FeatureConf(name='target feature path', file_path=target_file_path)

    def _train_run(self, **entity):
        assert "dataset" in entity.keys()
        self.dtype_inference(dataset=entity["dataset"])

        return self.feature_configure, self.target_feature_configure

    def ftype_inference(self, dataset: Bunch):
        data = dataset.data

        for index, column in enumerate(data.columns):
            if self.feature_configure.feature_dict[column].dtype == 'string':
                self.feature_configure.feature_dict[column].ftype = 'category'
            elif self.feature_configure.feature_dict[column].dtype == 'float64':
                self.feature_configure.feature_dict[column].ftype = 'numerical'
            else:
                assert self.feature_configure.feature_dict[column].dtype == 'int64'

                categorical_detect = len(pd.unique(data[column]))/data[column].shape[0]
                if categorical_detect < self.categorical_threshold:
                    self.feature_configure.feature_dict[column].ftype = 'category'
                else:
                    self.feature_configure.feature_dict[column].ftype = 'numerical'
        return self.feature_configure

    def dtype_inference(self, dataset: Bunch):

        data = dataset.data
        target = dataset.target

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

            self.feature_configure.add_item_type(column_name=column, feature_item_conf=feature_item_configure)

        for label_index, label in enumerate(target):
            feature_item_configure = FeatureItemConf()

            if self.task_name == 'regression':
                assert target[label].dtypes == 'float64' and target[label].isna().sum() == 0

            if self.task_name == 'classification':
                assert target[label].dtypes == 'int64'

            feature_item_configure.name = label
            feature_item_configure.index = label_index
            feature_item_configure.dtype = target[label].dtypes
            self.feature_configure.add_item_type(column_name=label, feature_item_conf=feature_item_configure)

        return self.feature_configure, self.target_feature_configure
