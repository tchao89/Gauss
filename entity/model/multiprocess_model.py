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

from utils.base import get_current_memory_gb
from utils.bunch import Bunch
from utils.Logger import logger
from utils.feature_index_exec import generate_feature_index


class MultiprocessModelWrapper(ModelWrapper, ABC):
    """
    This object is a base class for all machine learning model used multiprocess.
    """

    def __init__(self, **params):
        super().__init__(
            name=params["name"],
            model_path=params["model_path"],
            model_config_root=params["model_config_root"],
            feature_config_root=params["feature_config_root"],
            task_name=params["task_name"],
            train_flag=params["train_flag"]
        )

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
            "Reading base dataset, "
            "with current memory usage: {:.2f} GiB".format(
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

        return {"data": dataset.data, "target": dataset.target.flatten()}

    def update_feature_conf(self, feature_conf):
        self._feature_conf = feature_conf
        self._feature_list = generate_feature_index(feature_conf=self._feature_conf)
        assert self._feature_list is not None
        return self._feature_list
