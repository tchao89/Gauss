"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab
"""
import re
import copy

import numpy as np
import pandas as pd

from entity.dataset.base_dataset import BaseDataset
from gauss.type_inference.base_type_inference import BaseTypeInference
from entity.feature_configuration.feature_config import FeatureItemConf, FeatureConf

from utils.Logger import logger
from utils.constant_values import ConstantValues
from utils.yaml_exec import yaml_read
from utils.yaml_exec import yaml_write


class PlainTypeInference(BaseTypeInference):
    """
    1、如果存在某一列的特征是50%数字和50%的字母，这种id特征也是合理的，需要进行处理。
    2、typeinference中，需要将噪声数据全部重置为缺失值。
    """
    def __init__(self, **params):
        """
        TypeInference object can just change dataset, abnormal data in dataset will improve in PlainDataClear object.
        :param name:
        :param task_name:
        :param train_flag:
        :param source_file_path:
        :param final_file_path:
        :param final_file_prefix:
        """

        super(PlainTypeInference, self).__init__(
            name=params["name"],
            train_flag=params["train_flag"],
            task_name=params["task_name"],
            source_file_path=params["source_file_path"],
            final_file_path=params["final_file_path"],
            final_file_prefix=params["final_file_prefix"]
        )

        assert params[ConstantValues.task_name] in [ConstantValues.binary_classification,
                                                    ConstantValues.multiclass_classification,
                                                    ConstantValues.regression]

        self._task_name = params["task_name"]
        self.ftype_list = []
        self.dtype_list = []

        self.epsilon = 0.00001
        self.dtype_threshold = 0.95
        self.categorical_threshold = 0.01

        if params["source_file_path"] is not None:
            self.init_feature_configure = FeatureConf(name="source feature path", file_path=params["source_file_path"])
            self.init_feature_configure.parse(method="user")
        else:
            self.init_feature_configure = None

        self.final_feature_configure = FeatureConf(name='target feature path', file_path=params["final_file_path"])

    def _train_run(self, **entity):
        assert "train_dataset" in entity.keys()

        self.dtype_inference(dataset=entity["train_dataset"])

        self.ftype_inference(dataset=entity["train_dataset"])

        self.target_check(dataset=entity["train_dataset"])

        # self._check_init_final_conf()
        self.final_configure_generation()

    def _increment_run(self, **entity):
        self._predict_run(**entity)

    def _predict_run(self, **entity):
        # just detect error in test dataset.
        assert "infer_dataset" in entity.keys()
        conf = yaml_read(yaml_file=self._final_file_path)

        for col in entity["infer_dataset"].get_dataset().data.columns:
            assert col in list(conf)

    def _string_column_selector(self, feature_name: str):
        if self.init_feature_configure is not None \
                and self.init_feature_configure.feature_dict.get(feature_name) is not None \
                and self.init_feature_configure.feature_dict.get(feature_name).dtype == 'string':

            self.final_feature_configure.feature_dict[feature_name].dtype = 'string'

    def _datetime_column_selector(self, feature_name: str, dataset: pd.DataFrame):

        def datetime_map(x):
            x = str(x)

            if re.search(r"(\d{4}.\d{1,2}.\d{1,2})", x):
                return True
            else:
                return False

        if self.init_feature_configure is not None \
                and self.init_feature_configure.feature_dict.get(feature_name) is not None \
                and self.init_feature_configure.feature_dict[feature_name].ftype == 'datetime' \
                and self.init_feature_configure.feature_dict[feature_name].dtype == 'string':

            column_unique = list(set(dataset[feature_name]))

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

            self._datetime_column_selector(feature_name=column, dataset=data)

        return self.final_feature_configure

    def dtype_inference(self, dataset: BaseDataset):

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)

        data_dtypes = data.dtypes
        for col_index, column in enumerate(data):

            feature_item_configure = FeatureItemConf()
            feature_item_configure.name = column
            feature_item_configure.index = col_index
            if "int" in str(data_dtypes[col_index]):
                feature_item_configure.dtype = "int64"

            elif "float" in str(data_dtypes[col_index]):
                int_count = 0

                for item in data[column]:
                    if not np.isnan(item) and abs(item - int(item)) < self.epsilon:
                        int_count += 1

                if int_count + data[column].isna().sum() == data[column].shape[0]:
                    feature_item_configure.dtype = "int64"
                    dataset.need_data_clear = True

                else:
                    feature_item_configure.dtype = "float64"

            elif data_dtypes[col_index] == 'object' or 'category':

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
                    dataset.need_data_clear = True

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
            self._string_column_selector(feature_name=column)

        return self.final_feature_configure

    def target_check(self, dataset: BaseDataset):
        target = dataset.get_dataset().target

        # check if target columns is illegal.
        for label_index, label in enumerate(target):

            if self._task_name == ConstantValues.regression:
                assert "float" in str(target[label].dtypes) and target[label].isna().sum() == 0

            if self._task_name == ConstantValues.binary_classification:
                if "float" in str(target[label].dtypes):
                    target[label] = target[label].astype("int64")
                assert "int" in str(target[label].dtypes) or "object" in str(target[label].dtypes)

    def _check_init_final_conf(self):
        if self.init_feature_configure is None:
            return

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
                logger.info(item[0] + " feature dose not exist in type inference.")

    def final_configure_generation(self):
        yaml_dict = {}

        for item in self.final_feature_configure.feature_dict.items():
            item_dict = {"name": item[1].name, "index": item[1].index, "dtype": item[1].dtype, "ftype": item[1].ftype, "size": item[1].size}
            yaml_dict[item[0]] = item_dict

        yaml_write(yaml_dict=yaml_dict, yaml_file=self._final_file_path)
