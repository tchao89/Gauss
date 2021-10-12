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
from utils.constant_values import ConstantValues
from utils.feature_name_exec import generate_feature_list, generate_categorical_list


class ModelWrapper(Entity):
    """
    This object is a base class for all machine learning
    model which doesn't use multiprocess.
    """

    def __init__(self, **params):
        self._model_root_path = params[ConstantValues.model_root_path]
        self._feature_config_root = os.path.join(
            params[ConstantValues.model_root_path],
            ConstantValues.feature_configure
        )

        self._model_config_root = os.path.join(
            params[ConstantValues.model_root_path],
            ConstantValues.model_parameters
        )

        self._model_save_root = os.path.join(
            params[ConstantValues.model_root_path],
            ConstantValues.model_save
        )

        self._task_name = params[ConstantValues.task_name]
        self._train_flag = params[ConstantValues.train_flag]

        assert isinstance(params[ConstantValues.metric_eval_used_flag], bool), \
            "Value: metric_eval_used_flag must be bool type, but get {} instead.".format(
                type(params[ConstantValues.metric_eval_used_flag])
            )
        self._metric_eval_used_flag = params[ConstantValues.metric_eval_used_flag]

        assert isinstance(params[ConstantValues.use_weight_flag], bool), \
            "Value: use_weight_flag must be bool type, but get {} instead.".format(
                type(params[ConstantValues.use_weight_flag])
            )

        self._use_weight_flag = params[ConstantValues.use_weight_flag]

        # model_config is a dict containing all features and these attributes used in the model.
        # This dict will write to yaml file.
        # This parameters contains model parameters and preprocessing flags.
        self._model_config = None

        self._model = None
        self._target_names = None

        # MetricResult object.
        self._val_metric_result = None
        self._train_metric_result = None

        self._val_loss_result = None
        self._train_loss_result = None

        # input parameters from auto machine learning.
        self._model_params = None
        # FeatureConf entity object, offering features attributes.
        self._feature_conf = None
        # feature list
        self._feature_list = None
        self._categorical_list = None

        # Update best model and parameters.
        self._best_model = None
        self._best_val_metric_result = None
        self._best_model_params = None
        self._best_feature_list = None

        # recording all metric results.
        self.metric_history = []

        super().__init__(
            name=params[ConstantValues.name],
        )

    @property
    def val_best_metric_result(self):
        """
        Get best metric result in validation set.
        :return: MetricResult
        """
        return self._best_val_metric_result

    @abc.abstractmethod
    def _initialize_model(self):
        """
        This method is designed to initialize model.
        Any model which needs to initialize can override this method.
        :return: None
        """

    @classmethod
    def _check_bunch(cls, dataset: Bunch):
        keys = ConstantValues.dataset_items
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

        if self._best_val_metric_result is None:
            self._best_val_metric_result = MetricResult(
                name=self._val_metric_result.name,
                metric_name=self._val_metric_result.metric_name,
                result=self._val_metric_result.result,
                optimize_mode=self._val_metric_result.optimize_mode
            )

        if self._best_feature_list is None:
            self._best_feature_list = self._feature_list

        # this value is used for test program.
        self.metric_history.append(self._val_metric_result.result)
        if self._best_val_metric_result.__cmp__(self._val_metric_result) < 0:
            self._best_model = self._model
            self._best_model_params = self._model_params
            self._best_val_metric_result = MetricResult(
                name=self._val_metric_result.name,
                metric_name=self._val_metric_result.metric_name,
                result=self._val_metric_result.result,
                optimize_mode=self._val_metric_result.optimize_mode
            )
            self._best_feature_list = self._feature_list

        self.update_best()

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
    def increment(self, increment_dataset: BaseDataset, **entity):
        """
        incremental training model
        :param increment_dataset: BaseDataset object, incremental dataset
        :param entity: dict object, including other entity, such as Metric... etc.
        :return: None
        """

    @abc.abstractmethod
    def binary_train(self, init_model_path: str, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        """
        Training model, and this method doesn't contain evaluation.
        :param init_model_path:
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
             metric: BaseMetric,
             **entity):
        """
        Evaluate after each training.
        :param train_dataset: BaseDataset object
        :param val_dataset: BaseDataset object
        :param metric: BaseMetric object
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
    def val_metric(self):
        """
        Get validation metric.
        :return: MetricResult
        """
        return self._val_metric_result

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

    @property
    def model(self):
        """
        Get trained model.
        :return: a certainty model object.
        """
        return self._model

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

    def preprocess(self):
        """
        This method is used to implement Normalization, Standardization, Ont hot encoding which need
        self._train_flag parameters, and this operator needs model_config dict.
        :return: None
        """
        if self._train_flag:
            self._train_preprocess()
        else:
            self._predict_preprocess()

    @abc.abstractmethod
    def _train_preprocess(self):
        """
        preprocessing in training model.
        :return: None
        """

    @abc.abstractmethod
    def _predict_preprocess(self):
        """
        Preprocessing in inference.
        :return: None
        """

    @abc.abstractmethod
    def _loss_func(self, *params):
        """
        This method is used to customize loss function, and the
        input parameters and return value are decided by model.
        Usually, these parameters are needed:
        :param y_prob: np.ndarray, the probabilities of model output.
        :param y_true: np.ndarray, the ture label of dataset.
        :return: this value is decided by model, and this value
        is usually loss value, grad value or hess value.
        """

    @abc.abstractmethod
    def _eval_func(self, *params):
        """
        This method is used to customize metric function and the
        input parameters and return value are decided by model.
        Usually, these parameters are needed:
        :param y_prob: np.ndarray, the probabilities of model output.
        :param y_true: np.ndarray, the ture label of dataset.
        :return: this value is decided by model, and this value
        is usually loss value, grad value or hess value.
        """
