# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic Inc. All rights reserved.
# Authors: Lab

import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index

from utils.bunch import Bunch
from entity.dataset.base_dataset import BaseDataset

class SequenceDataset(BaseDataset):
    """Dataset for loading sequence shape time series data.
    """

    REG = "regression"
    CLS = "classification"
    MUL = "multi"
    UNI = "unique"

    def __init__(self, **params):
        """
        :param period_sep: delimeter between identify period data.
        :param fea_sep: delimeter between feature columns in a period.
        :param label_sep: delimeter between feature columns and labels in a period.
        :param has_feature_name: boolean value, True for including feature names in 1st
            row, False for not.
        """
        super(SequenceDataset, self).__init__(
            name=params["name"], 
            data_path=params["data_path"] \
                if params.get("data_path") else None,
            task_type=params["task_type"], 
            target_name=None, 
            memory_only=params["memory_only"] \
                 if params.get("memory_only") else True
        )
        self._filepath = self._data_path
        self._period_sep = params["period_seq"] \
            if params.get("period_seq") else ";"
        self._fea_sep = params["fea_seq"] \
            if params.get("fea_seq") else ","
        self._label_sep = params["label_seq"] \
            if params.get("label_seq") else "\t"
        self._has_feature_name = params["has_feature_name"] \
            if params.get("has_feature_name") else False
        
        self._miss_label = False
        self._val_start = None
        self.need_data_clear = False

        if not params.get("data_pair"):
            self._bunch = Bunch()
        else:
            self._bunch = params["data_pair"]

    def __repr__(self):
        content = "data:\n {data} target:\n {target} steps:\n {steps}".format(
            data=self._bunch.data,
            target=self._bunch.target,
            steps=self._bunch.steps
            )
        return content
     

    def build_dataset(self):
        file = open(self._filepath)
        self._read_from_file(file)
        file.close()

    def _read_from_file(self, file):
        line = file.readline()
        
        data = []
        labels = []
        label_index = []
        time_steps = []
        label_name = ["label"]
        stepper = 0

        if self._has_feature_name:
            fea_names, label_name = self._strip_and_split(line, self._label_sep)
            fea_names = self._strip_and_split(fea_names, self._fea_sep)
            label_name = [label_name.strip()]
            line = file.readline()

        while line:
            period_data, period_labels = self._feature_label_split(line)
            step_length = len(period_data)
            flag = True

            for step_data in period_data:
                if not self._has_feature_name and flag:                    
                    fea_names = [str(i) for i in range(len(step_data))]
                    flag = False
                if self._multi_fea:
                    data.append(self._strip_and_split(step_data, self._fea_sep))
                else:
                    data.append(step_data)

            if self._dataset_type == self.MUL:
                for idx, step_label in period_labels:
                    labels.append(step_label)
                    cur_idx = int(idx + np.array(time_steps[:stepper]).sum())
                    label_index.append(cur_idx)
                    stepper += 1
            elif self._dataset_type == self.UNI:
                labels += period_labels

            time_steps.append(step_length)
            line = file.readline()

        self._bunch.data = pd.DataFrame(data=data, columns=fea_names)
        self._bunch.target = pd.DataFrame(data=labels, columns=label_name)
        self._bunch.steps = pd.DataFrame(data=time_steps, columns=["steps"])
        self._bunch.feature_names = self._bunch.data.columns
        if self._dataset_type == self.MUL:
            self._bunch.indies = pd.DataFrame(data=label_index, columns=["indies"])

        print(self._bunch)

    def _feature_label_split(self, line):
        """Split feature columns and label columns to separete contents.

        And set the dataset type to `multi` or `unique` by the number of labels.
            1. `multi` means m to m, each sample has a label value. 
            2. `unique` means m to 1, a time period including m steps data has a single label.
        Meanwhile, m to m dataset also include a specific situation, couple lables may missed
        in a period, that will clarified by the attribute `miss_label`.
        """
        data, label = self._strip_and_split(line, self._label_sep)
        self._dataset_type = self.MUL if self._fea_sep in label else self.UNI
        self._multi_fea = True if self._fea_sep in data else False

        data = self._strip_and_split(data, self._period_sep)
        label = self._strip_and_split(label, self._period_sep)
        if self._dataset_type == self.MUL:
            self._miss_label = True if len(label) != len(data) else False
            label = [self._strip_and_split(x, self._fea_sep) for x in label]
            if self._task_type == self.REG:
                label = [(int(value[0]), float(value[1])) for value in label]
            else:
                label = [(int(value[0]), value[1]) for value in label]
        return data, label

    def _strip_and_split(self, sample, delimiter=None):
        sample = sample.strip()
        sample = sample.split(delimiter)
        return sample

    def load_data(self):
        self.build_dataset()

    def get_dataset(self):
        return self._bunch

    def split(self, val_start=0.8):
        if self._val_start is None:
            self._val_start = int(val_start * len(self._bunch.steps))
        bunch = self._bunch

        count = bunch.steps.iloc[self._val_start:, :].sum().values[0]
        
        valset = bunch.data.iloc[-count:, :]
        bunch.data = bunch.data.iloc[:-count, :]
        
        if self._dataset_type == "multi":
            valset_target = bunch.target.iloc[-count:, :]
            bunch.target = bunch.target.iloc[:-count, :]
        else:
            valset_target = bunch.target.iloc[self._val_start:, :]
            bunch.target = bunch.target.iloc[:self._val_start, :]

        valset_steps = bunch.steps.iloc[self._val_start:, :]
        bunch.steps = bunch.steps.iloc[:self._val_start, :]

        train = [bunch.data, bunch.target, bunch.steps]
        val = [valset, valset_target, valset_steps]
        self._reset_index(train+val)

        data_pair = Bunch(
            data=valset, target=valset_target, 
            steps=valset_steps,
            feature_names=bunch.feature_names
            )
        return SequenceDataset(
            name="seq_valset",
            task_type=self._task_type,
            data_pair=data_pair
        )
        
    def union(self, valset):
        bunch = self._bunch
        self._val_start = len(bunch.steps)
        count = bunch.steps.iloc[self._val_start:, :].sum().values[0]

        bunch.data = pd.concat([bunch.data, valset.get_dataset().data], axis=0)
        bunch.target = pd.concat([bunch.target, valset.get_dataset().target], axis=0)
        bunch.steps = pd.concat([bunch.steps, valset.get_dataset().steps], axis=0)

        bunch.data.reset_index(drop=True, inplace=True)
        bunch.target.reset_index(drop=True, inplace=True)
        bunch.steps.reset_index(drop=True, inplace=True)

    def feature_choose(self, feature_list):
        pass

    def _reset_index(self, args):
        for arg in args:
            arg.reset_index(drop=True, inplace=True)


    @property
    def data(self):
        return self._bunch.data

    @property
    def labels(self):
        return self._bunch.target

    @property
    def steps(self):
        return self._bunch.steps
    
    @property
    def dataset_type(self):
        return self._dataset_type

    @property
    def task_type(self):
        return self._task_type

    @property
    def columns(self):
        return self.data.columns

    @property
    def need_data_clean(self):
        return self.need_data_clear

    @need_data_clean.setter
    def need_data_clean(self, value):
        self.need_data_clear = value
