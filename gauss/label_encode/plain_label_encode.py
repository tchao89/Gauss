"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import shelve

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

from entity.dataset.base_dataset import BaseDataset
from gauss.label_encode.base_label_encode import BaseLabelEncode

from utils.Logger import logger
from utils.base import get_current_memory_gb
from utils.constant_values import ConstantValues
from utils.reduce_data import reduce_data
from utils.yaml_exec import yaml_read
from utils.yaml_exec import yaml_write


class PlainLabelEncode(BaseLabelEncode):
    """
    BaseLabelEncode Object.
    """

    def __init__(self, **params):

        super().__init__(
            name=params[ConstantValues.name],
            train_flag=params[ConstantValues.train_flag],
            enable=params[ConstantValues.enable],
            task_name=params[ConstantValues.task_name],
            feature_configure_path=params[ConstantValues.feature_configure_path]
        )

        self.__final_file_path = params["final_file_path"]

        self.__feature_configure = None

        self.__label_encoding_configure_path = params["label_encoding_configure_path"]
        self.__label_encoding = {}

        if self._task_name == ConstantValues.regression:
            self.__regression_label_switch = params[ConstantValues.label_switch_type]
        else:
            self.__regression_label_switch = None

        if self._train_flag == ConstantValues.train:
            self.__dataset_weight = params["dataset_weight"]
        else:
            self.__dataset_weight = None

    def _train_run(self, **entity):
        assert "train_dataset" in entity.keys()
        dataset = entity["train_dataset"]

        self.__load_dataset_configure()

        logger.info("Starting label encoding, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        self.__encode_label(dataset=dataset)
        self.__switch_label(switch_type=self.__regression_label_switch,
                            dataset=dataset)
        self.__reset_dataset_attributes(dataset=dataset)

        logger.info("Label encoding serialize, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        self.__serialize_label_encoding()
        self.__generate_final_configure()

    def _increment_run(self, **entity):
        assert ConstantValues.increment_dataset in entity.keys()
        dataset = entity[ConstantValues.increment_dataset]

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)

        target = dataset.get_dataset().target

        feature_names = dataset.get_dataset().feature_names
        target_names = dataset.get_dataset().target_names

        self.__feature_configure = yaml_read(yaml_file=self.__final_file_path)

        with shelve.open(self.__label_encoding_configure_path) as shelve_open:
            le_model_list = shelve_open['label_encoding']
            self.__label_encoding = le_model_list

            for col in feature_names:
                if not isinstance(self.__feature_configure, dict):
                    raise TypeError(
                        "Value: self.__feature_configure is not a correct data type, type: {}".format(
                            type(self.__feature_configure)
                        ))

                # transform features
                if self.__feature_configure[col]['ftype'] == "category" or \
                        self.__feature_configure[col]['ftype'] == "bool":
                    assert le_model_list.get(col)
                    le_model = le_model_list[col]

                    label_dict = dict(zip(le_model.classes_, le_model.transform(le_model.classes_)))
                    status_list = data[col].unique()

                    for item in status_list:
                        if label_dict.get(item) is None:
                            logger.info(
                                "feature: " + str(col) + " has an abnormal value (unseen by label encoding): " + str(
                                    item))
                            raise ValueError("feature: " + str(
                                col) + " has an abnormal value (unseen by label encoding): " + str(item))

                    data[col] = le_model.transform(data[col])

            # transform labels
            for col in target_names:
                if self._task_name == ConstantValues.binary_classification or \
                        self._task_name == ConstantValues.multiclass_classification:
                    assert le_model_list.get(col)
                    le_model = le_model_list[col]

                    label_dict = dict(zip(le_model.classes_, le_model.transform(le_model.classes_)))
                    status_list = target[col].unique()

                    for item in status_list:
                        if label_dict.get(item) is None:
                            logger.info(
                                "feature: " + str(col) + " has an abnormal value (unseen by label encoding): " + str(
                                    item))
                            raise ValueError("feature: " + str(
                                col) + " has an abnormal value (unseen by label encoding): " + str(item))

                    target[col] = le_model.transform(target[col])
                else:
                    assert le_model_list.get("switch")
                    trans_func = le_model_list["switch"]["func"]
                    trans_params = le_model_list["switch"]["params"]
                    target[col] = trans_func(target[col], *trans_params)

    def _predict_run(self, **entity):
        assert "infer_dataset" in entity.keys()
        dataset = entity['infer_dataset']

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)
        feature_names = dataset.get_dataset().feature_names

        self.__feature_configure = yaml_read(yaml_file=self.__final_file_path)

        with shelve.open(self.__label_encoding_configure_path) as shelve_open:
            le_model_list = shelve_open['label_encoding']

            for col in feature_names:
                if not isinstance(self.__feature_configure, dict):
                    raise TypeError(
                        "Value: self.__feature_configure is not a correct data type, type: {}".format(
                            type(self.__feature_configure)
                        ))

                if self.__feature_configure[col]['ftype'] == "category" \
                        or self.__feature_configure[col]['ftype'] == "bool":
                    assert le_model_list.get(col)
                    le_model = le_model_list[col]

                    label_dict = dict(zip(le_model.classes_, le_model.transform(le_model.classes_)))
                    status_list = data[col].unique()

                    for item in status_list:
                        try:
                            le_model.transform([item])
                        except ValueError:
                            logger.info(
                                "feature: " + str(col) + " has an abnormal value (unseen by label encoding): " + str(
                                    item))
                            logger.info("Label dict: {}".format(label_dict))
                            raise ValueError("feature: " + str(
                                col) + " has an abnormal value (unseen by label encoding): " + str(item))

                    data[col] = le_model.transform(data[col])

    def __encode_label(self, dataset: BaseDataset):
        data = dataset.get_dataset().data
        feature_names = dataset.get_dataset().feature_names

        target = dataset.get_dataset().target
        target_names = dataset.get_dataset().target_names
        for feature in feature_names:
            if self.__feature_configure[feature]['ftype'] == 'category' or \
                    self.__feature_configure[feature]['ftype'] == 'bool':
                item_label_encoding = LabelEncoder()
                item_label_encoding_model = item_label_encoding.fit(data[feature])
                self.__label_encoding[feature] = item_label_encoding_model

                data[feature] = item_label_encoding_model.transform(data[feature])

        if self._task_name == ConstantValues.binary_classification \
                or self._task_name == ConstantValues.multiclass_classification:
            for label in target_names:
                item_label_encoding = LabelEncoder()
                item_label_encoding_model = item_label_encoding.fit(target[label])
                self.__label_encoding[label] = item_label_encoding_model

                target[label] = item_label_encoding_model.transform(target[label])
        else:
            assert self._task_name == ConstantValues.regression
        logger.info(
            "Label encoding finished, starting to reduce dataframe and save memory, " + "with current memory usage: %.2f GiB",
            get_current_memory_gb()["memory_usage"])

    def __reset_dataset_attributes(self, dataset: BaseDataset):
        target_names = dataset.get_dataset().target_names

        encoding_weight = None
        encoding_proportion = {}
        if self._task_name == ConstantValues.binary_classification \
                or self._task_name == ConstantValues.multiclass_classification:
            for label in target_names:
                proportion = {}
                if self.__dataset_weight:
                    weight = {}
                    encoding_weight = {}
                    for label_value, label_weight in self.__dataset_weight[label].copy().items():
                        """
                        Using dict.copy() can avoid RuntimeError: dictionary changed size during iteration.
                        """
                        [encoding_value] = self.__label_encoding[label].transform([label_value])
                        logger.info(
                            "Original value: {} has been encoded to value: {}".format(label_value, encoding_value))
                        weight[encoding_value] = label_weight
                    encoding_weight[label] = weight

                for label_value, label_num in dataset.get_dataset().proportion[label].copy().items():
                    """
                    Using dict.copy() can avoid RuntimeError: dictionary changed size during iteration.
                    """
                    [encoding_value] = self.__label_encoding[label].transform([label_value])
                    logger.info("Original value: {} has been encoded to value: {}".format(label_value,
                                                                                          encoding_value))
                    proportion[encoding_value] = label_num
                encoding_proportion[label] = proportion

            if dataset.get_dataset().dataset_weight is None:
                dataset.get_dataset().dataset_weight = encoding_weight

            dataset.get_dataset().proportion = encoding_proportion

    def __serialize_label_encoding(self):
        # 序列化label encoding模型字典
        with shelve.open(self.__label_encoding_configure_path) as shelve_open:
            shelve_open['label_encoding'] = self.__label_encoding

    def __load_dataset_configure(self):
        self.__feature_configure = yaml_read(self._feature_configure_path)

    def __generate_final_configure(self):
        yaml_write(yaml_dict=self.__feature_configure, yaml_file=self.__final_file_path)

    @property
    def dataset_weight(self):
        return self.__dataset_weight

    def __switch_label(self, switch_type: str, dataset: BaseDataset):
        if self._task_name == ConstantValues.regression:
            target = dataset.get_dataset().target
            target_names = dataset.get_dataset().target_names
            for label in target_names:
                if label not in target.columns:
                    raise ValueError(
                        "label: {} is not in target names: {}.".format(
                            label, target.columns))
                if switch_type == "log":
                    target[label] = np.log(target[label])
                    switch_func = np.log
                    params = ()
                elif switch_type == "exp":
                    target[label] = np.exp(target[label])
                    switch_func = np.exp
                    params = ()
                elif switch_type == "pow":
                    target[label] = np.power(target[label], 0.5)
                    switch_func = np.power
                    params = (0.5,)
                elif switch_type == "scale":
                    target[label] = scale(target[label])
                    switch_func = scale
                    params = ()
                elif switch_type is None:
                    def switch_func():
                        pass

                    switch_func = switch_func
                    params = ()
                else:
                    raise ValueError(
                        "switch type: {} is not in switch types: {}.".format(
                            switch_type, ConstantValues.switch_types))
                self.__label_encoding["switch"] = {"func": switch_func, "params": params}
        else:
            return None
