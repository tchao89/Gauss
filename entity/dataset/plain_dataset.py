"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab
"""
from __future__ import annotations

import os
import string
from typing import List

import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file

from utils.bunch import Bunch
from entity.dataset.base_dataset import BaseDataset
from utils.Logger import logger
from utils.constant_values import ConstantValues
from utils.reduce_data import reduce_data


class PlaintextDataset(BaseDataset):
    """loading raw data to PlaintextDataset object.
    Further operation could be applied directly to current dataSet after the construction function
    used.
    Dataset can split by call `split()` function when need a validation set, and `union()` will merge
    train set and validation set in vertical.
    Features of current dataset can be eliminated by `feature_choose()` function, both index and
    feature name are accepted.
    """

    def __init__(self, **params):
        """
        Two kinds of raw data supported:
            1. The first is read file whose format is an option in `.csv`, `.txt`,and
        `.libsvm`, data after processing will wrapped into a `Bunch` object which contains
        `data` and `target`, meanwhile `feature_names` and `target_name` also can be a
        content when exist.
            2. Pass a `data_package` wrapped by Bunch to the construct function, `data` and
        `target` must provided at least.
        ======
        :param name: A string to represent module's name.
        :param data_path:  A string or `dict`; if string, it can be directly used by load data
            function, otherwise `train_dataset` and `val_dataset` must be the key of the dict.
        :param data_package: default is None, must be filled if data_path not applied.
        :param task_type: A string which is an option between `classification` or `regression`.
        :param target_name: A `list` containing label names in string format.
        :param memory_only: a boolean value, true for memory, false for others, default True.
        """
        for item in [ConstantValues.data_package,
                     ConstantValues.data_path,
                     ConstantValues.target_names,
                     ConstantValues.memory_only,
                     ConstantValues.data_file_type]:
            if params.get(item) is None:
                params[item] = None

        super().__init__(
            params[ConstantValues.name],
            params[ConstantValues.data_path],
            params[ConstantValues.task_name],
            params[ConstantValues.train_flag],
            params[ConstantValues.memory_only]
        )

        if bool(params[ConstantValues.data_path]) == bool(params[ConstantValues.data_package]):
            raise AttributeError("data_path or data_package must be provided.")

        self._data_package = params[ConstantValues.data_package]

        self._target_names = params[ConstantValues.target_names]
        self.__type_doc = params[ConstantValues.data_file_type]

        assert isinstance(self._target_names, List) or self._target_names is None, "Value: target_name is {}".format(
            self._target_names)

        self._default_print_size = 5
        self._bunch = None

        if params.get(ConstantValues.weight_column_flag):
            assert isinstance(params[ConstantValues.weight_column_flag], bool)
            if params[ConstantValues.weight_column_name]:
                assert isinstance(params[ConstantValues.weight_column_name], list)
                self._weight_column_names = params[ConstantValues.weight_column_name]

            self._weight_column_flag = params[ConstantValues.weight_column_flag]
        else:
            self._weight_column_flag = False
            self._weight_column_names = None

        # mark start point of validation set in all dataset, if just one data file offers, start point will calculate
        # by train_test_split = 0.3, and if train data file and validation file offer, start point will calculate
        # by the length of validation dataset.
        self._val_start = None
        # This value is a bool value, and true means plaindataset has missing values and need to clear.
        self._need_data_clear = False

        if params[ConstantValues.data_path] is not None \
                and isinstance(params[ConstantValues.data_path], str):
            self._column_name_flag = params[ConstantValues.column_name_flag]

            assert isinstance(self._column_name_flag, bool)
            self._bunch = self.load_data()
        else:
            self._bunch = self._data_package

    def __repr__(self):
        data = self._bunch.data
        target = self._bunch.target
        df = pd.concat((data, target), axis=1)
        return str(df.head(self._default_print_size)) + str(df.info())

    def get_dataset(self):
        return self._bunch

    def set_dataset(self, data_package):
        self._bunch = data_package
        return self

    def load_data(self, data_path=None):
        if data_path is not None:
            self._data_path = data_path

        if not os.path.isfile(self._data_path):
            raise TypeError("<{path}> is not a valid file.".format(
                path=self._data_path
            ))
        if self.__type_doc is None:
            self.__type_doc = self._data_path.split(".")[-1]

        if self.__type_doc not in ["csv", "libsvm", "txt"]:
            raise TypeError(
                "Unsupported file, excepted option in `csv`, `libsvm`, "
                "`txt`, {} received.".format(self.__type_doc)
            )

        data = None
        target = None
        feature_names = None
        target_names = None
        weight = None

        if self.__type_doc == "csv":

            try:
                data, target, feature_names, target_names, weight = self.load_csv()
            except IOError:
                logger.info("File path does not exist.")
            finally:
                logger.info(".csv file has been converted to Bunch object.")

            self._bunch = Bunch(data=data,
                                target=target,
                                target_names=target_names,
                                feature_names=feature_names,
                                dataset_weight=weight)

        elif self.__type_doc == 'libsvm':
            if self._column_name_flag:
                raise ValueError(
                    "Value: column_name_flag should be False, but get {} instead.".format(self._column_name_flag))
            try:
                data, target = self.load_libsvm()
            except ValueError:
                raise ValueError("Dataset file type is not correct, and it's not a libsvm file.")
            finally:
                logger.info(".libsvm file has been converted to Bunch object.")

            _, data, target = self._convert_data_dataframe(data=data,
                                                           target=target)

            if self._weight_column_flag is True:
                assert self._weight_column_names == "-1", \
                    "When type of dataset file is libsvm, " \
                    "value: self._weight_column_name should be set -1."

                weight = data.iloc[:, -1]
                data.drop(data.columns[-1], axis=1, inplace=True)
            else:
                weight = None

            self._bunch = Bunch(
                data=data,
                target=target,
                dataset_weight=weight
            )

            self._bunch.target_names = ["target_" + string.ascii_uppercase[index]
                                        for index, _ in enumerate(target)]
            self._bunch.feature_names = ["feature_" + str(index) for index, _ in enumerate(data)]
            data.columns = self._bunch.feature_names
            target.columns = self._bunch.target_names

        elif self.__type_doc == 'txt':
            if self._column_name_flag:
                raise ValueError(
                    "Value: column_name_flag should be False, but get {} instead.".format(self._column_name_flag))
            try:
                data, target = self.load_txt()
            except ValueError:
                raise ValueError("Dataset file type is not correct, and it's not a txt file.")
            finally:
                logger.info(".txt file has been converted to Bunch object.")

            _, data, target = self._convert_data_dataframe(data=data,
                                                           target=target)

            if self._weight_column_flag is True:
                assert self._weight_column_names == "-1", \
                    "When type of dataset file is txt, " \
                    "value: self._weight_column_name should be set -1."
                weight = data.iloc[:, -1]
                data.drop(data.columns[-1], axis=1, inplace=True)
            else:
                weight = None

            self._bunch = Bunch(
                data=data,
                target=target,
                dataset_weight=weight
            )

            self._bunch.target_names = ["target_" + string.ascii_uppercase[index]
                                        for index, _ in enumerate(target)]
            self._bunch.feature_names = ["feature_" + str(index) for index, _ in enumerate(data)]
            data.columns = self._bunch.feature_names
            target.columns = self._bunch.target_names

        else:
            raise TypeError("File type can not be accepted.")
        if self._train_flag == ConstantValues.train:
            self.__set_proportion()
        return self._bunch

    def __set_proportion(self):
        if self._train_flag == ConstantValues.train:
            if self._bunch is None:
                raise ValueError("Dataset has not been loaded.")
            if self._bunch.get("proportion"):
                raise ValueError("Value: self._proportion must be empty.")
            if not isinstance(self._bunch, Bunch):
                raise AttributeError("Value: self._bunch must be type: Bunch, "
                                     "but get {} instead.".format(type(self._bunch)))
            target = self._bunch.target
            target_names = self._bunch.target_names[0]

            count = 0
            proportion_dict = {}
            label_class_dict = {}

            for index, value in target[target_names].value_counts().iteritems():
                proportion_dict[index] = value
                count += 1
            label_class_dict[target_names] = count
            self._bunch.label_class = label_class_dict
            self._bunch.proportion = {target_names: proportion_dict}
        else:
            self._bunch.label_class = None
            self._bunch.proportion = None

    def load_csv(self):
        data = reduce_data(data_path=self._data_path,
                           column_name_flag=self._column_name_flag)
        if self._column_name_flag:
            if self._weight_column_flag is True:

                for weight_name in self._weight_column_names:
                    if weight_name not in data.columns:
                        raise ValueError("Column: {} doesn't exist in dataset file.".format(self._weight_column_names))

                weight = data[self._weight_column_names]
                data.drop(self._weight_column_names, axis=1, inplace=True)
            else:
                weight = None
            if self._target_names is not None:
                target = data[self._target_names]
                data = data.drop(self._target_names, axis=1)
            else:
                target = None
            feature_names = data.columns
            target_names = self._target_names
        else:
            target_columns = []
            generated_columns = list(data.columns)
            if self._weight_column_flag is True:
                if self._weight_column_names:
                    weight_columns = []
                    for weight_index in self._weight_column_names:
                        weight_columns.append(generated_columns[weight_index])

                    weight = data[weight_columns]
                    data.drop(weight_columns, axis=1, inplace=True)
                else:
                    weight = None
            else:
                weight = None

            if self._target_names is not None:
                for target_index in self._target_names:
                    target_columns.append(generated_columns[target_index])
                target_columns = list(set(target_columns))
                target = data.iloc[:, target_columns]
                data.drop(target_columns, axis=1, inplace=True)
                target_names = ["target_" + string.ascii_uppercase[index]
                                for index, _ in enumerate(target)]
                target.columns = target_names
            else:
                target = None
            feature_names = ["feature_" + str(index) for index, _ in enumerate(data)]
            data.columns = feature_names
            target_names = self._target_names
        return data, target, feature_names, target_names, weight

    def load_libsvm(self):
        data, target = load_svmlight_file(self._data_path)
        data = data.toarray()

        return data, target

    def load_txt(self):
        target_index = 0
        data = []
        target = []

        with open(self._data_path, 'r') as file:
            lines = file.readlines()

            for index, line_content in enumerate(lines):
                data_index = []
                line_content = line_content.split(' ')

                for column, item in enumerate(line_content):
                    if column != target_index:
                        data_index.append(item)
                    else:
                        target.append(item)

                data_index = list(map(np.float64, data_index))
                data.append(data_index)

            data = np.asarray(data, dtype=np.float64)
            target = list(map(int, target))
            target = np.asarray(target, dtype=int)

            return data, target

    def wc_count(self):
        import subprocess
        out = subprocess.getoutput("wc -l %s" % self._data_path)
        return int(out.split()[0])

    def get_target_name(self):
        return self._target_names

    @classmethod
    def _convert_data_dataframe(cls, data, target,
                                feature_names=None, target_names=None):

        data_df = pd.DataFrame(data, columns=feature_names)
        target_df = pd.DataFrame(target, columns=target_names)
        combined_df = pd.concat([data_df, target_df], axis=1)

        return combined_df, data_df, target_df

    def feature_choose(self, feature_list: list, use_index_flag: bool):
        assert isinstance(use_index_flag, bool)
        if use_index_flag:
            return self._bunch.data.iloc[:, feature_list]
        else:
            return self._bunch.data.loc[:, feature_list]

    # dataset is a PlainDataset object
    def union(self, val_dataset: PlaintextDataset):
        """ This method is used for concatenating train dataset and validation dataset.Merge train set and
        validation set in vertical, this procedure will operated on train set.
        example:
            trainSet = PlaintextDataset(...)
            validationSet = trainSet.split()
            trainSet.union(validationSet)

        :return: Plaindataset
        """
        self._val_start = self._bunch.target.shape[0]
        if self._bunch.data.shape[1] != val_dataset.get_dataset().data.shape[1]:
            raise ValueError("Shape of train dataset is not consistent with shape of validation dataset.")

        self._bunch.data = pd.concat([self._bunch.data, val_dataset.get_dataset().data], axis=0)

        assert self._bunch.target.shape[1] == val_dataset.get_dataset().target.shape[1]
        self._bunch.target = pd.concat([self._bunch.target, val_dataset.get_dataset().target], axis=0)

        self._bunch.data = self._bunch.data.reset_index(drop=True)
        self._bunch.target = self._bunch.target.reset_index(drop=True)

        if self._bunch.get("feature_names") is not None and val_dataset.get_dataset().get("feature_names") is not None:
            for item in self._bunch.feature_names:
                assert item in val_dataset.get_dataset().feature_names

        if self._bunch.get("target_names") is not None and val_dataset.get_dataset().get("target_names") is not None:
            for item in self._bunch.target_names:
                assert item in val_dataset.get_dataset().target_names

    def split(self, val_start: float = 0.8):
        """split a validation set from train set.
        :param val_start: ration of sample count as validation set.
        :return: plaindataset object containing validation set.
        """
        assert self._bunch is not None

        for key in self._bunch.keys():
            assert key in ConstantValues.dataset_items

        if self._val_start is None:
            self._val_start = int(val_start * self._bunch.data.shape[0])

        val_data = self._bunch.data.iloc[self._val_start:, :]
        self._bunch.data = self._bunch.data.iloc[:self._val_start, :]

        val_target = self._bunch.target.iloc[self._val_start:]
        self._bunch.target = self._bunch.target.iloc[:self._val_start]

        self._bunch.data.reset_index(drop=True, inplace=True)
        self._bunch.target.reset_index(drop=True, inplace=True)

        val_data = val_data.reset_index(drop=True)
        val_target = val_target.reset_index(drop=True)

        data_package = Bunch(data=val_data, target=val_target)

        if self._weight_column_flag:
            val_dataset_weight = self._bunch.dataset_weight.iloc[self._val_start:]
            self._bunch.dataset_weight = self._bunch.dataset_weight.iloc[:self._val_start]
            val_dataset_weight = val_dataset_weight.reset_index(drop=True)
            self._bunch.dataset_weight.reset_index(drop=True, inplace=True)
            data_package.dataset_weight = val_dataset_weight

        if "feature_names" in self._bunch.keys():
            data_package.target_names = self._bunch.target_names
            data_package.feature_names = self._bunch.feature_names

        if ConstantValues.dataset_weight in self._bunch.keys():
            data_package.dataset_weight = self._bunch.dataset_weight

        data_package.proportion = self._bunch.proportion

        return PlaintextDataset(name=self._name,
                                task_name=self._task_name,
                                train_flag=ConstantValues.train,
                                weight_column_flag=self._weight_column_flag,
                                weight_column_name=self._weight_column_names,
                                data_package=data_package)

    @property
    def need_data_clear(self):
        return self._need_data_clear

    @need_data_clear.setter
    def need_data_clear(self, data_clear: bool):
        self._need_data_clear = data_clear

    @property
    def target_names(self):
        return self._target_names

    @property
    def default_print_size(self):
        return self._default_print_size
