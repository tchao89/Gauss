"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
Template used to create customized model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

from entity.model.single_process_model import SingleProcessModelWrapper
from entity.dataset.base_dataset import BaseDataset
from entity.metrics.base_metric import BaseMetric


class GaussUdfModel(SingleProcessModelWrapper):
    """
    lightgbm object.
    """
    # This flag is bool values, if true,
    # dataset must use data clear component when training this model.
    need_data_clear = False

    def __init__(self, **params):

        super().__init__(
            name=params["name"],
            model_path=params["model_path"],
            model_config_root=params["model_config_root"],
            feature_config_root=params["feature_config_root"],
            task_name=params["task_name"],
            train_flag=params["train_flag"]
        )

        self.__model_file_name = self.name + ".txt"
        self.__model_config_file_name = self.name + ".yaml"
        self.__feature_config_file_name = self.name + ".yaml"

    def __repr__(self):
        pass

    def __load_data(self, dataset: BaseDataset, val_dataset: BaseDataset = None):
        """

        :param val_dataset:
        :param dataset:
        :return: lgb.Dataset
        """
        pass

    def _initialize_model(self):
        pass

    def train(self,
              train_dataset: BaseDataset,
              val_dataset: BaseDataset,
              **entity
              ):
        pass

    def predict(self, infer_dataset: BaseDataset, **entity):
        pass

    def preprocess(self):
        pass

    def _train_preprocess(self):
        pass

    def _predict_process(self):
        pass

    def eval(self,
             train_dataset: BaseDataset,
             val_dataset: BaseDataset,
             metrics: BaseMetric,
             **entity
             ):
        """

        :param train_dataset: BaseDataset object, used to get training metric and loss.
        :param val_dataset: BaseDataset object, used to get validation metric and loss.
        :param metrics: BaseMetric object, used to calculate metrics.
        :param entity: dict object, including other entity.
        :return: None
        """
        pass

    def model_save(self, model_path=None):
        pass

    def set_weight(self):
        """
        This method can set weight for different label.
        :return: None
        """

    def update_best(self):
        """
        Do not need to operate.
        :return:
        """

    def set_best(self):
        """
        Do not need to operate.
        :return:
        """
