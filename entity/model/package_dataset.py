"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab
"""
import pandas as pd

from utils.Logger import logger
from utils.base import get_current_memory_gb
from utils.bunch import Bunch


class PackageDataset:
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

        data = dataset.get_dataset().data

        assert isinstance(data, pd.DataFrame)
        assert isinstance(use_weight_flag, bool)

        for feature in data.columns:
            if feature in categorical_list:
                data[feature] = data[feature].astype("category")

        if feature_list is not None:
            # check type of feature_choose
            data = dataset.feature_choose(feature_list, use_index_flag=False)
            target = dataset.get_dataset().target

            data_package = Bunch(
                data=data,
                target=target,
                target_names=dataset.get_dataset().target_names,
                dataset_weight=self.__dataset_weight,
                categorical_list=categorical_list
            )

            dataset_bunch = data_package
        else:
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
            return self.__load_dataset(
                self,
                dataset=dataset_bunch,
                train_flag=train_flag,
            )

        return self.__load_dataset(
            self,
            dataset=Bunch(data=dataset.data.values),
            train_flag=train_flag)
