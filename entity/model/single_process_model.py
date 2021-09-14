"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab
"""
from __future__ import annotations

import copy
from abc import ABC

from entity.dataset.base_dataset import BaseDataset
from entity.model.model import ModelWrapper

from utils.Logger import logger
from utils.base import get_current_memory_gb
from utils.bunch import Bunch
from utils.feature_name_exec import feature_list_generator


class SingleProcessModelWrapper(ModelWrapper, ABC):
    """
    This object is a base class for all machine learning model used multiprocess.
    """

    def __init__(self, **params):
        super().__init__(
            name=params["name"],
            model_root_path=params["model_root_path"],
            task_name=params["task_name"],
            train_flag=params["train_flag"]
        )

    def update_feature_conf(self, feature_conf=None):
        """
        This method will update feature conf and transfer feature configure to feature list.
        :param feature_conf: FeatureConfig object
        :return:
        """
        if feature_conf is not None:
            self._feature_conf = feature_conf
            self._feature_list = feature_list_generator(feature_conf=self._feature_conf)
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

            data_pair = Bunch(
                data=data,
                target=target,
                target_names=dataset.get_dataset().target_names
            )

            dataset = copy.deepcopy(dataset).set_dataset(data_pair=data_pair)

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
            return {"data": dataset.data.values,
                    "target": dataset.target.values.flatten()}

        return {"data": dataset.data.values}


class choose_features:
    def __init__(self, func):
        self.__load_dataset = func

    def __call__(self, *args, **kwargs):

        dataset = kwargs.get("dataset")
        check_bunch = kwargs.get("check_bunch")
        feature_list = kwargs.get("feature_list")
        train_flag = kwargs.get("train_flag")

        if feature_list is not None:
            data = dataset.feature_choose(feature_list)
            target = dataset.get_dataset().target

            data_pair = Bunch(
                data=data,
                target=target,
                target_names=dataset.get_dataset().target_names
            )

            dataset = copy.deepcopy(dataset).set_dataset(data_pair=data_pair)

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
        check_bunch(dataset=dataset)

        logger.info(
            "Construct lgb.Dataset object in load_data method, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        if train_flag:
            return self.__load_dataset(
                self,
                dataset={"data": dataset.data.values,
                         "target": dataset.target.values.flatten()},
                train_flag=train_flag)

        return self.__load_dataset(
            self,
            dataset={"data": dataset.data.values},
            train_flag=train_flag)
