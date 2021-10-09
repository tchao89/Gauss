"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab
"""
from __future__ import annotations

import copy
from statistics import harmonic_mean
from abc import ABC

import pandas as pd

from entity.dataset.base_dataset import BaseDataset
from entity.model.model import ModelWrapper

from utils.Logger import logger
from utils.base import get_current_memory_gb
from utils.bunch import Bunch
from utils.feature_name_exec import generate_feature_list
from utils.feature_name_exec import generate_categorical_list
from utils.constant_values import ConstantValues


class SingleProcessModelWrapper(ModelWrapper, ABC):
    """
    This object is a base class for all machine learning model used multiprocess.
    """

    def __init__(self, **params):
        super().__init__(
            name=params["name"],
            model_root_path=params["model_root_path"],
            task_name=params["task_name"],
            train_flag=params["train_flag"],
            metric_eval_used_flag=params["metric_eval_used_flag"],
            use_weight_flag=params["use_weight_flag"]
        )

    def update_feature_conf(self, feature_conf=None):
        """
        This method will update feature conf and transfer feature configure to feature list.
        :param feature_conf: FeatureConfig object
        :return:
        """
        if feature_conf is not None:
            self._feature_conf = feature_conf
            self._feature_list = generate_feature_list(feature_conf=self._feature_conf)
            self._categorical_list = generate_categorical_list(feature_conf=self._feature_conf)
            assert self._feature_list is not None
            return self._feature_list

        return None

    def _generate_sub_dataset(self, dataset: BaseDataset):
        """
        Generate a new BaseDataset object by self._feature_list.
        :param dataset: BaseDataset
        :return: BaseDataset
        """
        if self._feature_list is not None:
            data = dataset.feature_choose(self._feature_list)
            target = dataset.get_dataset().target

            data_package = Bunch(
                data=data,
                target=target,
                target_names=dataset.get_dataset().target_names
            )

            dataset = copy.deepcopy(dataset).set_dataset(data_package=data_package)

        logger.info(
            "Reading base dataset, with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        dataset = dataset.get_dataset()

        logger.info(
            "Check base dataset, with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )
        self._check_bunch(dataset=dataset)

        logger.info(
            "Construct lgb.Dataset object in load_data method, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        if self._train_flag:
            return {"data": dataset.data,
                    "target": dataset.target,
                    "target_names": dataset.target_names}

        return {"data": dataset.data.values,
                "target_names": dataset.target_names}


class choose_features:
    def __init__(self, func):
        self.__load_dataset = func
        self.__dataset_weight = None

    def __call__(self, *args, **kwargs):
        dataset = kwargs.get("dataset")
        check_bunch = kwargs.get("check_bunch")
        feature_list = kwargs.get("feature_list")
        train_flag = kwargs.get("train_flag")
        categorical_list = kwargs.get("categorical_list")
        use_weight_flag = kwargs.get("use_weight_flag")
        task_name = kwargs.get("task_name")

        data = dataset.get_dataset().data
        weight = dataset.get_dataset().dataset_weight

        assert isinstance(data, pd.DataFrame)
        assert isinstance(use_weight_flag, bool)

        if use_weight_flag is True:
            if dataset.get_dataset().dataset_weight is None:
                self.__set_weight(dataset=dataset, task_name=task_name)
            else:
                logger.info("Weight column is in dataset, weight generation method will not start.")

        for feature in data.columns:
            if feature in categorical_list:
                data[feature] = data[feature].astype("category")

        if feature_list is not None:
            data = dataset.feature_choose(feature_list)
            target = dataset.get_dataset().target

            data_package = Bunch(
                data=data,
                target=target,
                target_names=dataset.get_dataset().target_names,
                dataset_weight=self.__dataset_weight,
                categorical_list=categorical_list
            )

            if use_weight_flag is True and weight is not None:
                assert isinstance(weight, (pd.DataFrame, pd.Series))
                data_package.dataset_weight = weight

            dataset = copy.deepcopy(dataset).set_dataset(data_package=data_package)

        logger.info(
            "Reading base dataset, with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        dataset_bunch = dataset.get_dataset()

        logger.info(
            "Check base dataset, with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )
        check_bunch(dataset=dataset_bunch)

        logger.info(
            "Construct lgb.Dataset object in load_data method, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        if train_flag:
            self.__reset_params()
            return self.__load_dataset(
                self,
                dataset=dataset,
                train_flag=train_flag,
            )

        return self.__load_dataset(
            self,
            dataset={"data": dataset.data.values},
            train_flag=train_flag)

    def __set_weight(self, dataset: BaseDataset, task_name: str):
        dataset_bunch = dataset.get_dataset()
        proportion = dataset_bunch.proportion
        dataset_weight = dataset_bunch.dataset_weight

        if not dataset_weight:
            if task_name == ConstantValues.binary_classification or ConstantValues.multiclass_classification:
                dataset_weight = {}
                for target_name, proportion_dict in proportion.items():
                    weight = {}
                    harmonic_value = harmonic_mean(proportion_dict.values())
                    for label_value, label_num in proportion_dict.items():
                        weight[label_value] = harmonic_value / label_num
                    dataset_weight[target_name] = weight
            else:
                dataset_weight["target_name"] = None
        self.__dataset_weight = dataset_weight

    def __reset_params(self):
        self.__dataset_weight = None
