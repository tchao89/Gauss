"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab
"""
from __future__ import annotations

import os
import abc

from entity.entity import Entity
from entity.dataset.base_dataset import BaseDataset
from entity.metrics.base_metric import BaseMetric
from entity.metrics.base_metric import MetricResult

from utils.bunch import Bunch


class ModelWrapper(Entity):
    """
    This object is a base class for all machine learning
    model which doesn't use multiprocess.
    """
    def __init__(self, **params):
        self._model_root_path = params["model_root_path"]
        self._feature_config_root = os.path.join(
            params["model_root_path"],
            "feature_config"
        )

        self._model_config_root = os.path.join(
            params["model_root_path"],
            "model_parameters"
        )

        self._model_save_root = os.path.join(
            params["model_root_path"],
            "model_save"
        )

        self._task_name = params["task_name"]
        self._train_flag = params["train_flag"]

        # model_config is a dict containing all features and these attributes used in the model.
        # This dict will write to yaml file.
        # This parameters contains model parameters and preprocessing flags.
        self._model_config = None

        self._model = None

        # MetricResult object.
        self._val_metrics_result = None
        self._train_metrics_result = None

        self._val_loss_result = None
        self._train_loss_result = None

        # input parameters from auto machine learning.
        self._model_params = None
        # FeatureConf entity object, offering features attributes.
        self._feature_conf = None
        # feature list
        self._feature_list = None

        # Update best model and parameters.
        self._best_model = None
        self._best_val_metrics_result = None
        self._best_model_params = None
        self._best_feature_list = None

        # recording all metric results.
        self.metrics_history = []

        super().__init__(
            name=params["name"],
        )

    @property
    def val_best_metric_result(self):
        """
        Get best metric result in validation set.
        :return: MetricResult
        """
        return self._best_val_metrics_result

    @abc.abstractmethod
    def _generate_sub_dataset(self, dataset: BaseDataset):
        pass

    @abc.abstractmethod
    def _initialize_model(self):
        """
        This method is designed to initialize model.
        Any model which needs to initialize can override this method.
        :return: None
        """

    @classmethod
    def _check_bunch(cls, dataset: Bunch):
        keys = ["data", "target", "feature_names", "target_names", "generated_feature_names"]
        for key in dataset.keys():
            assert key in keys

    def update_best_model(self):
        """
        This method will update model, model parameters and
        validation metric result in each training.
        :return: None
        """
        if self._best_model is None:
            self._best_model = self._model

        if self._best_model_params is None:
            self._best_model_params = self._model_params

        if self._best_val_metrics_result is None:
            self._best_val_metrics_result = MetricResult(
                name=self._val_metrics_result.name,
                result=self._val_metrics_result.result,
                optimize_mode=self._val_metrics_result.optimize_mode
            )

        if self._best_feature_list is None:
            self._best_feature_list = self._feature_list

        # this value is used for test program.
        self.metrics_history.append(self._val_metrics_result.result)
        if self._best_val_metrics_result.__cmp__(self._val_metrics_result) < 0:
            self._best_model = self._model
            self._best_model_params = self._model_params
            self._best_val_metrics_result = MetricResult(
                name=self._val_metrics_result.name,
                result=self._val_metrics_result.result,
                optimize_mode=self._val_metrics_result.optimize_mode
            )
            self._best_feature_list = self._feature_list

        self.update_best()

    @abc.abstractmethod
    def update_best(self):
        """
        This method is designed to customize values to update_best_model()
        :return: None
        """

    @abc.abstractmethod
    def train(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        """
        Training model, and this method doesn't contain evaluation.
        :param train_dataset: BaseDataset object, training dataset.
        :param val_dataset: BaseDataset object, validation dataset.
        :param entity: dict object, including other entity, such as Metric... etc.
        :return:
        """

    @abc.abstractmethod
    def predict(self, infer_dataset: BaseDataset, **entity):
        """
        Predicting and get inference result.
        :param infer_dataset: BaseDataset object, evaluating dataset.
        :param entity: dict object, including BaseDataset, Metric... etc.
        :return: None
        """

    @abc.abstractmethod
    def eval(self,
             train_dataset: BaseDataset,
             val_dataset: BaseDataset,
             metrics: BaseMetric,
             **entity):
        """
        Evaluate after each training.
        :param train_dataset: BaseDataset object
        :param val_dataset: BaseDataset object
        :param metrics: BaseMetric object
        :param entity: dict object, including other entity object.
        :return: None
        """

    @property
    def train_metric(self):
        """
        Get train metric.
        :return: MetricResult
        """
        return self._val_loss_result

    @property
    def train_loss(self):
        """
        Get training loss.
        :return: MetricResult
        """
        return self._train_loss_result

    @property
    def val_loss(self):
        """
        Get validation loss.
        :return: MetricResult
        """
        return self._val_loss_result

    @property
    def val_metrics(self):
        """
        Get validation metrics.
        :return: MetricResult
        """
        return self._val_metrics_result

    @property
    def feature_list(self):
        """
        Get feature list.
        :return: list
        """
        return self._feature_list

    @property
    def model_params(self):
        """
        Get model parameters.
        :return: dict
        """
        return self._model_params

    @property
    def model_config(self):
        """
        Get model config.
        :return: dict
        """
        return self._model_config

    @model_config.setter
    def model_config(self, model_config: dict):
        """
        Set model config, and this method will be used in supervised feature selector object.
        :param model_config:
        :return: None
        """
        self._model_config = model_config

    @abc.abstractmethod
    def model_save(self):
        """
        Save model in model save root.
        :return: None
        """

    def update_params(self, **params):
        """
        update parameters, this method will be used in auto machine learning component.
        :param params:
        :return:
        """
        if self._model_params is None:
            self._model_params = {}

        self._model_params.update(params)

    @abc.abstractmethod
    def set_best(self):
        """
        This method is designed to customize values to set_best_model()
        :return:
        """

    def set_best_model(self):
        """
        # This method will convert self.best_object to self.object
        :return: None
        """
        self._model = self._best_model
        self._model_params = self._best_model_params
        self._feature_list = self._best_feature_list
        self.set_best()

    @abc.abstractmethod
    def preprocess(self):
        """
        This method is used to implement Normalization, Standardization, Ont hot encoding which need
        self._train_flag parameters, and this operator needs model_config dict.
        :return: None
        """
        if self._train_flag:
            self._train_preprocess()
        else:
            self._predict_process()

    @abc.abstractmethod
    def _train_preprocess(self):
        """
        preprocessing in training model.
        :return: None
        """

    @abc.abstractmethod
    def _predict_process(self):
        """
        Preprocessing in inference.
        :return: None
        """
