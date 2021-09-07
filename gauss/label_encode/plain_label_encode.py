"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import shelve

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from entity.dataset.base_dataset import BaseDataset
from gauss.label_encode.base_label_encode import BaseLabelEncode

from utils.Logger import logger
from utils.base import get_current_memory_gb, reduce_data
from utils.common_component import yaml_read, yaml_write


class PlainLabelEncode(BaseLabelEncode):
    """
    BaseLabelEncode Object.
    """
    def __init__(self, **params):

        super().__init__(
            name=params["name"],
            train_flag=params["train_flag"],
            enable=params["enable"],
            task_name=params["task_name"],
            feature_configure_path=params["feature_config_path"]
        )

        self.__final_file_path = params["final_file_path"]

        self.__feature_configure = None

        self.__label_encoding_configure_path = params["label_encoding_configure_path"]
        self.__label_encoding = {}

    def _train_run(self, **entity):
        assert "train_dataset" in entity.keys()
        dataset = entity["train_dataset"]

        self.__load_dataset_configure()

        logger.info("Starting label encoding, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        self.__encode_label(dataset=dataset)

        logger.info("Label encoding serialize, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        self.__serialize_label_encoding()
        self.__generate_final_configure()

    def _predict_run(self, **entity):
        assert "infer_dataset" in entity.keys()
        dataset = entity['infer_dataset']

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)

        target = dataset.get_dataset().target

        feature_names = dataset.get_dataset().feature_names
        target_names = dataset.get_dataset().target_names

        with shelve.open(self.__label_encoding_configure_path) as shelve_open:
            le_model_list = shelve_open['label_encoding']

            for col in feature_names:
                if self.__feature_configure[col]['ftype'] == "category" or self.__feature_configure[col]['ftype'] == "bool":
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

            for col in target_names:
                if self._task_name == "classification":
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

        if self._task_name == "classification":
            for label in target_names:
                item_label_encoding = LabelEncoder()
                item_label_encoding_model = item_label_encoding.fit(target[label])
                self.__label_encoding[label] = item_label_encoding_model

                target[label] = item_label_encoding_model.transform(target[label])

        logger.info("Label encoding finished, starting to reduce dataframe and save memory, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        reduce_data(dataframe=dataset.get_dataset().data)

    def __serialize_label_encoding(self):
        # 序列化label encoding模型字典
        with shelve.open(self.__label_encoding_configure_path) as shelve_open:
            shelve_open['label_encoding'] = self.__label_encoding

    def __load_dataset_configure(self):
        self.__feature_configure = yaml_read(self._feature_configure_path)

    def __generate_final_configure(self):
        yaml_write(yaml_dict=self.__feature_configure, yaml_file=self.__final_file_path)
