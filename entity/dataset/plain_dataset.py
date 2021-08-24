# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
from __future__ import annotations

import os
import csv
import pandas as pd
import numpy as np

from sklearn.datasets import load_svmlight_file

from utils.bunch import Bunch
from entity.dataset.base_dataset import BaseDataset
from utils.Logger import logger
from utils.base import reduce_data


class PlaintextDataset(BaseDataset):
    """loading raw data to PlaintextDataset object.

    Further operation could be applied directly to current datasett after the construction function
    used.
    Dataset can split by call `split()` function when need a validation set, and `union()` will merge 
    train set and validation set in vertical.
    Features of current dataset can be elimated by `feature_choose()` function, both index and 
    feature name are accepted.
    """

    def __init__(self, **params):
        """
        Two kinds of raw data supported: 
            1. The first is read file whose format is an option in `.csv`, `.txt`,and 
        `.libsvm`, data after processing will wrapped into a `Bunch` object which contains
        `data` and `target`, meanwhile `feature_names` and `target_name` also can be a 
        contenet when exist.
            2. Pass a `data_pair` wrapped by Bunch to the construct function, `data` and 
        `target` must provided at least. 
    
        ======    
        :param name: A string to represent module's name.
        :param data_path:  A string or `dict`; if string, it can be directly used by load data 
            function, otherwise `train_dataset` and `val_dataset` must be the key of the dict.
        :param data_pair: default is None, must be filled if data_path not applied.
        :param task_type: A string which is an option between `classification` or `regression`.
        :param target_name: A `list` containing label names in string format.
        :param memory_only: a boolean value, true for memory, false for others, default True.
        """
        for item in [
            "name", 
            "task_type", 
            "data_pair", 
            "data_path", 
            "target_name", 
            "memory_only"
            ]:
            if params.get(item) is None:
                params[item] = None

        if not params["data_path"] and not params["data_pair"]:
            raise AttributeError("data_path or data_pair must provided.")

        super(PlaintextDataset, self).__init__(
            name=params["name"], 
            data_path=params["data_path"], 
            task_type=params["task_type"], 
            target_name=params["target_name"], 
            memory_only=params["memory_only"]
        )
        
        self._data_pair = params["data_pair"]

        self._suffix = None
        self._val_start = None
        self._bunch = None
        self._need_data_clear = False

        if isinstance(self._data_path, str):
            self._bunch = self.load_data()
        elif isinstance(self._data_path, dict):
            if not (self._data_path.get("train_dataset") and self._data_path.get("val_dataset")):
                raise ValueError(
                    "data_path must include `train_dataset` and `val_dataset` when pass a dictionary."
                )
            self._bunch = Bunch(
                train_dataset=self.load_data(data_path=self._data_path["train_dataset"]),
                val_dataset=self.load_data(data_path=self._data_path["val_dataset"])
            )
        else:
            self._bunch = self._data_pair


    def __repr__(self):
        data = self._bunch.data
        target = self._bunch.target
        df = pd.concat((data, target), axis=1)
        return str(df.head()) + str(df.info())


    def load_data(self, data_path=None):
        if data_path is not None:
            self._data_path = data_path
            
        if not os.path.isfile(self._data_path):
            raise TypeError("<{path}> is not a valid file.".format(
                path = self._data_path
            ))

        self._suffix = os.path.splitext(self._data_path)[-1] 
        if self._suffix not in [".csv", ".libsvm", ".txt"]:
            raise TypeError(
                "Unsupported file, excepted option in `.csv`, `.libsvm`, `.txt`, <{suffix}> received."\
                .format(suffix=self._suffix)
            )

        data = None
        target = None
        feature_names = None
        target_name = None

        if self._suffix == ".csv":
            try:
                data, target, feature_names, target_name = self._load_mixed_csv()
            except IOError:
                logger.info("File not exist.")
        else: 
            if self._suffix == '.libsvm':
                data, target = self._load_libsvm()
            elif self._suffix == '.txt':
                data, target = self._load_txt()
            data, target = self._convert_to_dataframe(data=data, target=target)
        
        self._bunch = Bunch(data=data, target=target)
        if feature_names.all():
            self._bunch.feature_names = feature_names
            self._bunch.target_names = target_name
        return self._bunch

    def _load_mixed_csv(self):
        target = None
        target_name = None
        
        data = reduce_data(data_path=self._data_path)

        feature_names = data.columns
        self._row_size = data.shape[0]

        if self._target_name is not None:
            target = data.loc[:, self._target_name]
            data = data.drop(self._target_name, axis=1)
            feature_names = data.columns
            target_name = self._target_name
            self._column_size = data.shape[1] + target.shape[1]
        return data, target, feature_names, target_name

    def load_csv(self):
        """Loads data from csv_file_name.

        Returns
        -------
        data : Numpy array
            A 2D array with each row representing one sample and each column
            representing the features of a given sample.

        target : Numpy array
            A 1D array holding target variables for all the samples in `data.
            For example target[0] is the target variable for data[0].

        target_names : Numpy array
            A 1D array containing the names of the classifications. For example
            target_names[0] is the name of the target[0] class.
        """
        with open(self._data_path, 'r') as csv_file:

            data_file = csv.reader(csv_file)
            feature_names = next(data_file)
            target_location = -1

            try:
                target_location = feature_names.index(self._target_name)
                target_name = feature_names.pop(target_location)
            except IndexError:
                logger.info("Label is not exist.")
            assert target_name == self._target_name

            self._row_size = n_samples = self.wc_count() - 1
            self._column_size = n_features = len(feature_names)

            data = np.empty((n_samples, n_features))
            target = np.empty((n_samples,), dtype=int)

            for index, row in enumerate(data_file):
                label = row.pop(target_location)
                data[index] = np.asarray(row, dtype=np.float64)
                target[index] = np.asarray(label, dtype=int)

        return data, target, feature_names, target_name

    def _load_libsvm(self):
        data, target = load_svmlight_file(self._data_path)
        data = data.toarray()
        self._column_size = len(data[0]) + 1
        self._row_size = len(data)
        return data, target

    def _load_txt(self):
        data = []
        target = []
        target_index = 0

        with open(self._data_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line_contents = line.split(' ')
                target.append(line_contents.pop(target_index))
                data.append(list(map(np.float64, line_contents)))
            data = np.asarray(data, dtype=np.float64)
            target = np.asarray(target, dtype=np.int64)
            self._column_size = len(data[0]) + 1
            self._row_size = len(data)
        return data, target

    def _convert_to_dataframe(self, data, target,
                                feature_names=None, 
                                target_names=None):
        data_df = pd.DataFrame(data, columns=feature_names)
        target_df = pd.DataFrame(target, columns=target_names)
        return data_df, target_df
    
    def wc_count(self):
        import subprocess
        out = subprocess.getoutput("wc -l %s" % self._data_path)
        return int(out.split()[0])


    def get_dataset(self):
        return self._bunch
    
    def get_column_size(self):
        return self._column_size

    def get_row_size(self):
        return self._row_size

    def get_target_name(self):
        return self._target_name


    def feature_choose(self, feature_list):
        try:
            data = self._bunch.data.iloc[:, feature_list]
        except IndexError:
            logger.info("index method used.")
            data = self._bunch.data.loc[:, feature_list]
        return data

    def union(self, val_dataset: PlaintextDataset):
        """ Merge train set and validation set in vertical, this procedure will operated on train set.

        example:
            trainset = PlaintextDataset(...)
            valset = trainset.split()
            trainset.union(valset) 
        """
        bunch = self._bunch

        self._val_start = bunch.target.shape[0]

        assert bunch.data.shape[1] == val_dataset.get_dataset().data.shape[1]
        bunch.data = pd.concat([bunch.data, val_dataset.get_dataset().data], axis=0)

        assert bunch.target.shape[1] == val_dataset.get_dataset().target.shape[1]
        bunch.target = pd.concat([bunch.target, val_dataset.get_dataset().target], axis=0)

        bunch.data = bunch.data.reset_index(drop=True)
        bunch.target = bunch.target.reset_index(drop=True)

        if bunch.get("feature_names") is not None and val_dataset.get_dataset().get("feature_names") is not None:
            for item in bunch.feature_names:
                assert item in val_dataset.get_dataset().feature_names
        if bunch.get("target_names") is not None and val_dataset.get_dataset().get("target_names") is not None:
            for item in bunch.target_names:
                assert item in val_dataset.get_dataset().target_names

    def split(self, val_start: float = 0.8):
        """split a validation set from train set.

        :param val_start: ration of sample count as validation set.

        :return: plaindataset object containing validation set.
        """
        assert self._bunch is not None
        bunch = self._bunch

        for key in bunch.keys():
            assert key in ["data", "target", "feature_names", "target_names", "generated_feature_names"]

        if self._val_start is None:
            self._val_start = int(val_start * bunch.data.shape[0])

        val_data = bunch.data.iloc[self._val_start:, :]
        bunch.data = bunch.data.iloc[:self._val_start, :]

        val_target = bunch.target.iloc[self._val_start:]
        bunch.target = bunch.target.iloc[:self._val_start]

        bunch.data.reset_index(drop=True, inplace=True)
        bunch.target.reset_index(drop=True, inplace=True)

        val_data = val_data.reset_index(drop=True)
        val_target = val_target.reset_index(drop=True)

        data_pair = Bunch(data=val_data, target=val_target)
        if "feature_names" in bunch.keys():
            data_pair.target_names = bunch.target_names
            data_pair.feature_names = bunch.feature_names

        return PlaintextDataset(
            name="train_and_val_set",
            task_type=self._task_type,
            data_pair=data_pair
            )


    @property
    def need_data_clear(self):
        return self._need_data_clear

    @need_data_clear.setter
    def need_data_clear(self, data_clear_flag: bool):
        self._need_data_clear = data_clear_flag
