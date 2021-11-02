"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab
"""
from __future__ import annotations

import os
import string
from typing import List
from statistics import harmonic_mean

import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle

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
            params[ConstantValues.memory_only]
        )

        if bool(params[ConstantValues.data_path]) == bool(params[ConstantValues.data_package]):
            raise AttributeError("data_path or data_package must be provided.")

        self._data_package = params[ConstantValues.data_package]

        self.__column_index = None
        self._target_names = params[ConstantValues.target_names]
        self.__type_doc = params[ConstantValues.data_file_type]

        assert isinstance(self._target_names, List) or self._target_names is None, "Value: target_name is {}".format(
            self._target_names)

        self._default_print_size = 5
        self._bunch = None

        if self._name == ConstantValues.train_dataset or self._name == ConstantValues.val_dataset:
            self.__use_weight_flag = params[ConstantValues.use_weight_flag]
            self.__dataset_weight_dict = params[ConstantValues.dataset_weight_dict]
            if params.get(ConstantValues.weight_column_name):
                assert isinstance(params[ConstantValues.weight_column_name], list)
                self._weight_column_names = params[ConstantValues.weight_column_name]
            else:
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
        elif self.__type_doc == 'libsvm':
            try:
                data, target, feature_names, target_names, weight = self.load_libsvm()
            finally:
                logger.info(".libsvm file has been converted to Bunch object.")
        elif self.__type_doc == 'txt':
            try:
                data, target, feature_names, target_names, weight = self.load_txt()
            except ValueError:
                raise ValueError("Dataset file type is not correct, and it's not a txt file.")
            finally:
                logger.info(".txt file has been converted to Bunch object.")
        else:
            raise TypeError("File type can not be accepted.")

        self._bunch = Bunch(data=data,
                            target=target,
                            target_names=target_names,
                            feature_names=feature_names,
                            dataset_weight=weight)

        # All kinds of dataset will get column names when programming reaches here.
        if self._name == ConstantValues.train_dataset or self._name == ConstantValues.val_dataset:
            self.__set_proportion()
            self._bunch.data, self._bunch.target = shuffle(self._bunch.data, self._bunch.target)
            self.__set_weight()
        elif self._name == ConstantValues.increment_dataset:
            self._bunch.data, self._bunch.target = shuffle(self._bunch.data, self._bunch.target)
        else:
            assert self._name == ConstantValues.infer_dataset
        return self._bunch

    def __set_proportion(self):
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

    def __set_weight(self):
        self.__dict_set_weight()
        self.__auto_set_weight()

    def __auto_set_weight(self):
        if not self.__use_weight_flag:
            return None

        proportion = self._bunch.proportion
        dataset_weight = self._bunch.dataset_weight
        target = self._bunch.target
        # Dataset weight will be calculated.
        if dataset_weight is None:
            if self._task_name == ConstantValues.binary_classification or ConstantValues.multiclass_classification:
                dataset_weight = {}
                for target_name, proportion_dict in proportion.items():
                    weight = {}
                    harmonic_value = harmonic_mean(proportion_dict.values())
                    for label_value, label_num in proportion_dict.items():
                        weight[label_value] = harmonic_value / label_num

                    weight_col = [weight[item] for item in target[target_name].values]
                    dataset_weight[target_name] = weight_col
                self._bunch.dataset_weight = pd.DataFrame(dataset_weight)

    def __dict_set_weight(self):
        """
        # eg. {"target_A": {1: 1.9, -1: 1}}, {-1: {1: 1.9, -1: 1}},
        if label weight dict has been set, this weight will be used, otherwise weight will be set 1.
        Note: column name has been replaced, so key of dataset weight dict need updated.
        :return: None10 Minutes to cuDF and Dask-cuDF
        """
        if not self.__use_weight_flag:
            return None

        if not bool(self.__dataset_weight_dict):
            return None

        dataset_weight = self._bunch.dataset_weight
        target = self._bunch.target
        target_names = self._bunch.target_names

        # Dataset weight will be calculated.
        if dataset_weight is None:
            if self._task_name == ConstantValues.binary_classification or ConstantValues.multiclass_classification:
                dataset_weight = {}
                for target_name, weight_dict in self.__dataset_weight_dict.items():
                    if target_name not in target_names:
                        raise ValueError(
                            "Weight dict target_name: {} is not in target names: {}.".format(
                                target_name, target_names))
                    weight_col = [weight_dict[item] for item in target[target_name].values]
                    dataset_weight[target_name] = weight_col

                remained_target_names = list(set(target_names) - set(self.__dataset_weight_dict.keys()))
                if remained_target_names:
                    for target_name in remained_target_names:
                        weight_col = [1 for _ in target[target_name].values]
                        dataset_weight[target_name] = weight_col
                self._bunch.dataset_weight = pd.DataFrame(dataset_weight)

    def load_csv(self):
        data = reduce_data(data_path=self._data_path,
                           column_name_flag=self._column_name_flag)

        if bool(self._weight_column_names) and bool(self.__dataset_weight_dict):
            raise ValueError("Just one weight setting can be set.")

        if self._column_name_flag:
            if self._weight_column_names:

                for weight_name in self._weight_column_names:
                    if weight_name not in data.columns:
                        raise ValueError("Column: {} doesn't exist in dataset file.".format(self._weight_column_names))

                weight = data[self._weight_column_names]
                data.drop(self._weight_column_names, axis=1, inplace=True)
            else:
                weight = None

            if self._name in [ConstantValues.train_dataset,
                              ConstantValues.val_dataset,
                              ConstantValues.increment_dataset]:
                target = data[self._target_names]
                target_names = self._target_names
                data.drop(self._target_names, axis=1, inplace=True)
            else:
                target = None
                target_names = None
            feature_names = list(data.columns)
        else:
            target_columns = []
            self.__column_index = list(data.columns)

            if self._weight_column_names:
                if self._weight_column_names:
                    weight_columns = []
                    for weight_index in self._weight_column_names:
                        weight_columns.append(self.__column_index[weight_index])

                    weight = data[weight_columns]
                    data.drop(weight_columns, axis=1, inplace=True)
                else:
                    weight = None
            else:
                weight = None

            if self.__dataset_weight_dict:
                for index in self.__dataset_weight_dict.copy().keys():
                    self.__dataset_weight_dict[self.__column_index[index]] = self.__dataset_weight_dict[index]
                    self.__dataset_weight_dict.pop(index)

            if self._name in [ConstantValues.train_dataset,
                              ConstantValues.val_dataset,
                              ConstantValues.increment_dataset]:

                for target_index in self._target_names:
                    target_columns.append(self.__column_index[target_index])

                target_columns = list(set(target_columns))
                target = data.iloc[:, target_columns]

                data.drop(target_columns, axis=1, inplace=True)
                target_names = ["target_" + string.ascii_uppercase[index]
                                for index, _ in enumerate(target_columns)]

                target_dict = dict(zip(target_columns, target_names))

                if weight is not None:
                    for index, weight_name in enumerate(weight.columns):
                        weight.rename(columns={weight_name: target_names[index]},
                                      inplace=True)

                if self.__dataset_weight_dict:
                    for weight_index in self.__dataset_weight_dict.copy().keys():
                        if weight_index in target_columns:
                            self.__dataset_weight_dict[target_dict[weight_index]] = self.__dataset_weight_dict[weight_index]
                            self.__dataset_weight_dict.pop(weight_index)
                        else:
                            raise IndexError("Weight index: {} is not consistent with target index: {}.".format(
                                weight_index, target_columns))

                target.columns = target_names
            else:
                target_names = None
                target = None

            feature_names = ["feature_" + str(index) for index, _ in enumerate(data)]
            data.columns = feature_names
        return data, target, feature_names, target_names, weight

    def load_libsvm(self):
        data, target = load_svmlight_file(self._data_path)
        data = pd.DataFrame(data.toarray())
        target = pd.DataFrame(target)

        if bool(self._weight_column_names) and bool(self.__dataset_weight_dict):
            raise ValueError("Just one weight setting can be set.")

        if self._target_names is not None:
            raise ValueError("Value: target names should be None when loading libsvm file.")

        if self._column_name_flag:
            raise ValueError("Value: column name flag should be false when loading libsvm file.")
        else:
            target_columns = list(target.columns)
            self.__column_index = list(data.columns)

            if self._weight_column_names:
                if self._weight_column_names:
                    weight_columns = []
                    for weight_index in self._weight_column_names:
                        weight_columns.append(self.__column_index[weight_index])

                    weight = data[weight_columns]
                    data.drop(weight_columns, axis=1, inplace=True)
                else:
                    weight = None
            else:
                weight = None

            if self._name in [ConstantValues.train_dataset,
                              ConstantValues.val_dataset,
                              ConstantValues.increment_dataset]:
                target_names = ["target_" + string.ascii_uppercase[index]
                                for index, _ in enumerate(target_columns)]
                target_dict = dict(zip(target_columns, target_names))

                if weight is not None:
                    for index, weight_name in enumerate(weight.columns):
                        weight.rename(columns={weight_name: target_names[index]},
                                      inplace=True)

                if self.__dataset_weight_dict:
                    for weight_index in self.__dataset_weight_dict.copy().keys():
                        if weight_index in target_columns:
                            self.__dataset_weight_dict[target_dict[weight_index]] = self.__dataset_weight_dict[weight_index]
                            self.__dataset_weight_dict.pop(weight_index)
                        else:
                            raise IndexError("Weight index: {} is not consistent with target index: {}.".format(
                                weight_index, target_columns))

                target.columns = target_names
            else:
                target_names = None
                target = None

            feature_names = ["feature_" + str(index) for index, _ in enumerate(data)]
            data.columns = feature_names
        return data, target, feature_names, target_names, weight

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

            data = pd.DataFrame(np.asarray(data, dtype=np.float64))
            target = list(map(int, target))
            target = pd.DataFrame(np.asarray(target, dtype=int))

            if bool(self._weight_column_names) and bool(self.__dataset_weight_dict):
                raise ValueError("Just one weight setting can be set.")

            if self._target_names is not None:
                raise ValueError("Value: target names should be None when loading libsvm file.")

            if self._column_name_flag:
                raise ValueError("Value: column name flag should be false when loading libsvm file.")
            else:
                target_columns = list(target.columns)
                self.__column_index = list(data.columns)

                if self._weight_column_names:
                    if self._weight_column_names:
                        weight_columns = []
                        for weight_index in self._weight_column_names:
                            weight_columns.append(self.__column_index[weight_index])

                        weight = data[weight_columns]
                        data.drop(weight_columns, axis=1, inplace=True)
                    else:
                        weight = None
                else:
                    weight = None

                if self._name in [ConstantValues.train_dataset,
                                  ConstantValues.val_dataset,
                                  ConstantValues.increment_dataset]:
                    target_names = ["target_" + string.ascii_uppercase[index]
                                    for index, _ in enumerate(target_columns)]
                    target_dict = dict(zip(target_columns, target_names))

                    if weight is not None:
                        for index, weight_name in enumerate(weight.columns):
                            weight.rename(columns={weight_name: target_names[index]},
                                          inplace=True)

                    if self.__dataset_weight_dict:
                        for weight_index in self.__dataset_weight_dict.copy().keys():
                            if weight_index in target_columns:
                                self.__dataset_weight_dict[target_dict[weight_index]] = self.__dataset_weight_dict[
                                    weight_index]
                                self.__dataset_weight_dict.pop(weight_index)
                            else:
                                raise IndexError("Weight index: {} is not consistent with target index: {}.".format(
                                    weight_index, target_columns))

                    target.columns = target_names
                else:
                    target_names = None
                    target = None

                feature_names = ["feature_" + str(index) for index, _ in enumerate(data)]
                data.columns = feature_names
            return data, target, feature_names, target_names, weight

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

        if self._bunch.dataset_weight is not None:
            assert isinstance(self._bunch.dataset_weight, (pd.DataFrame, pd.Series))
            assert self._bunch.dataset_weight.shape[1] == val_dataset.get_dataset().dataset_weight.shape[1]
            self._bunch.dataset_weight = pd.concat([self._bunch.dataset_weight,
                                                    val_dataset.get_dataset().dataset_weight], axis=0)
            self._bunch.dataset_weight = self._bunch.dataset_weight.reset_index(drop=True)

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

        if self._bunch.dataset_weight is not None:
            assert isinstance(self._bunch.dataset_weight, (pd.DataFrame, pd.Series))
            val_dataset_weight = self._bunch.dataset_weight.iloc[self._val_start:]
            self._bunch.dataset_weight = self._bunch.dataset_weight.iloc[:self._val_start]
            val_dataset_weight = val_dataset_weight.reset_index(drop=True)
            self._bunch.dataset_weight.reset_index(drop=True, inplace=True)
            data_package.dataset_weight = val_dataset_weight
        else:
            data_package.dataset_weight = None

        if self._bunch.feature_names is not None:
            data_package.feature_names = self._bunch.feature_names
        else:
            data_package.feature_names = None

        if self._bunch.target_names is not None:
            data_package.target_names = self._bunch.target_names
        else:
            data_package.target_names = None

        if self._bunch.generated_feature_names is not None:
            data_package.generated_feature_names = self._bunch.generated_feature_names
        else:
            data_package.generated_feature_names = None

        data_package.proportion = self._bunch.proportion
        data_package.label_class = self._bunch.label_class

        return PlaintextDataset(name=ConstantValues.val_dataset,
                                task_name=self._task_name,
                                train_flag=ConstantValues.train,
                                use_weight_flag=self.__use_weight_flag,
                                weight_column_name=self._weight_column_names,
                                dataset_weight_dict=self.__dataset_weight_dict,
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
