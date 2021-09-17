# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic Inc. All rights reserved.
# Authors: Lab

import numpy as np
import pandas as pd
import tensorflow as tf

from utils.bunch import Bunch
from entity.dataset.base_dataset import BaseDataset

class SequenceDataset(BaseDataset):
    """Dataset for loading sequence shape time series data.

    Dataset .txt file to be loaded should strictly follow the format
    standard:

        1. A line of data stands for a period, which include data part
            and label part. M to M or M to 1 are supported. 

        2. A period can include single or couples time steps. 
        
        3. A time step can include single or couples features.
        
        4. Labels in M to M should contain time step indies and values.
        
        5. Values and labels should be seprated by `\\t`.
        
        6. Time steps should be seperated by `;`, this rule applies in both
            values and labels.
        
        7. Features should be seperated by `,`, same as rule 6.

    Parameters:
    -----------

    period_sep : delimeter between identify period data.

    fea_sep : delimeter between feature columns in a period.

    label_sep : delimeter between feature columns and labels in a period.

    has_feature_name : boolean value, True for including feature names in 1st
        row, False for not.

    Attributes:
    -----------

    label_mapping : many-to-many, or many-to-one.

    task_name : current dataset label type , `classification` or 
        `regression`.
    """

    REG = "regression"
    CLS = "classification"
    MUL = "multi"
    UNI = "unique"

    def __init__(self, **params):
        """
        Parameters:
        -----------
        period_sep : delimeter between identify period data.

        fea_sep : delimeter between feature columns in a period.
  
        label_sep : delimeter between feature columns and labels in a period.
  
        has_feature_name : boolean value, True for including feature names in 1st
            row, False for not.
        """
        super(SequenceDataset, self).__init__(
            name=params["name"], 
            data_path=params["data_path"] \
                if params.get("data_path") else None,
            task_name=params["task_name"], 
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
        # self._target_name = params["target_name"]

        self._label_mapping = None
        self._miss_label = False
        self._label_mask = None
        self._val_start = None
        self.need_data_clear = False
        self._label_type = self._label_type()

        if not params.get("data_pair"):
            self._bunch = Bunch()
        else:
            self._bunch = params["data_pair"]

    def __repr__(self):
        return str(self._bunch)
     
    def update_features(self, features: list, cate_fea):
        """update private attribute selected_features, which are actually
        needed features in current trail.
        """
        if self._target_name:
            self._selected_features = features + self._target_name
        else:
            self._selected_features = features
        self._categorical_features = cate_fea

    def load_dataset(self):
        """Read .txt file to SequenceDataset."""
        file = open(self._filepath)
        self._read_from_txt(file)
        file.close()

    def _read_from_txt(self, file):
        line = file.readline()
        data = []
        labels = []
        time_steps = []
        mask = []
        label_name = "label"

        if self._has_feature_name:
            fea_names, label_name = self._strip_and_split(line, self._label_sep)
            fea_names = self._strip_and_split(fea_names, self._fea_sep)
            label_name = label_name.strip()
            line = file.readline()

        while line:
            period_data, period_labels = self._feature_label_split(line)
            step_length = len(period_data)
            counter = 0
            flag = True

            for idx, step_data in enumerate(period_data):
                if not self._has_feature_name and flag:                    
                    fea_names = [str(i) for i in range(len(step_data))]
                    flag = False
                if self._multi_fea:
                    data.append(self._strip_and_split(step_data, self._fea_sep))
                else:
                    data.append(step_data)

                if self._label_mapping == self.MUL:
                    if self._miss_label:
                        try:
                            step_label_idx = period_labels[idx-counter][0]
                        except Exception:
                            step_label_idx = period_labels[-1][0]
                        
                        if step_label_idx == idx:
                            labels.append(period_labels[idx-counter][1])
                        else:
                            counter += 1
                            labels.append(np.nan)
                    else:
                        labels.append(period_labels[idx][1])
                elif self._label_mapping == self.UNI:
                    labels += period_labels
            
            time_steps.append(step_length)
            line = file.readline()

        labels = np.array(labels)
        if self._label_mapping == self.MUL:
            self._label_mask = (~np.isnan(labels)).astype(np.int32)
        np.nan_to_num(labels, nan=0, copy=False)

        self._bunch.data = pd.DataFrame(data=data, columns=fea_names)
        self._bunch.target = pd.Series(data=labels, name=label_name)
        self._bunch.steps = pd.Series(data=time_steps, name="steps")
        self._bunch.feature_names = self._bunch.data.columns.tolist()
        self._bunch.target_names = label_name
        

    def _feature_label_split(self, line):
        """Split feature columns and label columns to separete contents.

        And set the dataset type to `multi` or `unique` by the number of labels.

            1. `multi` means m to m, each sample has a label value.

            2. `unique` means m to 1, a time period including m steps data has a single label.

        Meanwhile, m to m dataset also include a specific situation, couple lables may missed
        in a period, that will clarified by the attribute `miss_label`.
        """
        data, label = self._strip_and_split(line, self._label_sep)
        self._label_mapping = self.MUL if self._fea_sep in label else self.UNI
        self._multi_fea = True if self._fea_sep in data else False

        data = self._strip_and_split(data, self._period_sep)
        label = self._strip_and_split(label, self._period_sep)
        if self._label_mapping == self.MUL:
            self._miss_label = True if len(label) != len(data) else False
            label = [self._strip_and_split(x, self._fea_sep) for x in label]
            if self._task_name == self.REG:
                label = [(int(idx), float(value)) for idx, value in label]
            else:
                label = [(int(idx), value) for idx, value in label]
        return data, label

    def _strip_and_split(self, sample, delimiter=None):
        sample = sample.strip()
        sample = sample.split(delimiter)
        return sample

    def split(self, val_start=0.8):
        if self._val_start is None:
            self._val_start = int(val_start * len(self._bunch.steps))
        bunch = self._bunch
        count = bunch.steps.iloc[self._val_start:, :].sum().values[0]        
        valset = bunch.data.iloc[-count:, :]
        bunch.data = bunch.data.iloc[:-count, :]
        
        if self._label_mapping == "multi":
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
            task_name=self._task_name,
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

    def _reset_index(self, args):
        for arg in args:
            arg.reset_index(drop=True, inplace=True)


    def build_tf_dataset(self, ) -> tf.data.Dataset:
        """Dataset for tf operations, like embedding, statistic calculation, etc.."""

        data = self._bunch.data
        
        pass 

    def _v_stack(self, dataset) -> pd.DataFrame:
        X = dataset.get_dataset().data
        y = dataset.get_dataset().target
        dataset = pd.concat((X, y), axis=1)
        return dataset
    
    # def _seq_dataset_gen(self):
    #     """Convert pd.DF to tf.Dataset."""
    #     bunch = self._bunch
    #     start_idx = 0
    #     end_idx = 0
    #     for step in bunch.steps.values:
    #         end_idx += step
    #         period_data = bunch.data.iloc[start_idx:end_idx,:].values
    #         period_label = bunch.target[start_idx:end_idx].values
    #         start_idx = end_idx
    #         yield period_data, period_label


    
    def load_data(self):
        self.build_dataset()

    def get_dataset(self):
        return self._bunch
        
    def feature_choose(self, feature_list):
        pass

    def _label_type(self):
        if self._task_name == self.REG:
            return tf.float32
        elif self._task_name == self.CLS:
            return tf.int32
            

    @property
    def label_mapping(self):
        return self._label_mapping

    @property
    def label_mask(self):
        return self._label_mask

    @property
    def task_name(self):
        return self._task_name

    @property
    def columns(self):
        return self.data.columns

    @property
    def need_data_clean(self):
        return self.need_data_clear

    @need_data_clean.setter
    def need_data_clean(self, value):
        self.need_data_clear = value
