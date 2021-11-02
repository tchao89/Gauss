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

    def __call__(self, *args, **kwargs):
        dataset = kwargs.get("dataset")
        check_bunch = kwargs.get("check_bunch")
        feature_list = kwargs.get("feature_list")
        train_flag = kwargs.get("train_flag")
        categorical_list = kwargs.get("categorical_list")

        data = dataset.get_dataset().data

        assert isinstance(data, pd.DataFrame)

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
                feature_names=dataset.get_dataset().feature_names,
                target_names=dataset.get_dataset().target_names,
                generated_feature_names=dataset.get_dataset().generated_feature_names,
                dataset_weight=dataset.get_dataset().dataset_weight,
                categorical_list=categorical_list,
                label_class=dataset.get_dataset().label_class,
                proportion=dataset.get_dataset().proportion
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
