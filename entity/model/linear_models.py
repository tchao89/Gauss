# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

import os
import shelve

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor

from entity.model.single_process_model import SingleProcessModelWrapper
from entity.dataset.base_dataset import BaseDataset
from entity.metrics.base_metric import BaseMetric, MetricResult
from utils.bunch import Bunch
from utils.base import mkdir


class GaussLinearModels(SingleProcessModelWrapper):
    need_data_clear = True

    def __init__(self, **params):
        super().__init__(
            params["name"],
            params["model_path"],
            params["task_name"],
            params["train_flag"]
        )

        self.model_file_name = self.name + ".txt"
        self.model_config_file_name = self.name + ".yaml"
        self.feature_config_file_name = self.name + ".yaml"

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

    @classmethod
    def __check_bunch(cls, dataset: Bunch):
        keys = ["data", "target", "feature_names", "target_names", "generated_feature_names"]
        for key in dataset.keys():
            assert key in keys

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

    def update_params(self, **params):
        if self._model_params is None:
            self._model_params = {}

        self._model_params.update(params)

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
