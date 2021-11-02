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
        super().__init__(
            name=params[ConstantValues.name],
        )

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

        if self._train_flag == ConstantValues.train:
            self._init_model_root = params[ConstantValues.init_model_root]
            if self._init_model_root is not None:
                self._init_model_path = os.path.join(self._init_model_root, self._name)
            else:
                self._init_model_path = None
        elif self._train_flag == ConstantValues.increment:
            self._decay_rate = params[ConstantValues.decay_rate]
        elif self._train_flag == ConstantValues.inference:
            self._increment_flag = params[ConstantValues.increment_flag]
            self._infer_result_type = params[ConstantValues.infer_result_type]
        else:
            raise ValueError("Value: train flag is invalid.")

        assert isinstance(params[ConstantValues.metric_eval_used_flag], bool), \
            "Value: metric_eval_used_flag must be bool type, but get {} instead.".format(
                type(params[ConstantValues.metric_eval_used_flag])
            )
        self._metric_eval_used_flag = params[ConstantValues.metric_eval_used_flag]

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
        self._model_file_name = None

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
            assert key in keys, "Value: {} is not in dataset items: {}.".format(key, keys)

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

        self._update_best()

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
    def _update_best(self):
        """
        This method is designed to customize values to update_best_model()
        :return: None
        """

    def run(self, **entity):
        if self._train_flag == ConstantValues.train:
            return self.train(**entity)
        elif self._train_flag == ConstantValues.inference:
            return self.inference(**entity)
        elif self._train_flag == ConstantValues.increment:
            return self.increment(**entity)
        else:
            raise ValueError("Value: train flag is invalid.")

    def train(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        dataset = train_dataset.get_dataset()
        self._check_bunch(dataset=dataset)
        if self._task_name in [ConstantValues.binary_classification,
                               ConstantValues.multiclass_classification,
                               ConstantValues.regression]:
            if self._task_name == ConstantValues.binary_classification:
                self._binary_train(train_dataset=train_dataset, val_dataset=val_dataset, **entity)
            elif self._task_name == ConstantValues.multiclass_classification:
                self._multiclass_train(train_dataset=train_dataset, val_dataset=val_dataset, **entity)
            elif self._task_name == ConstantValues.regression:
                self._regression_train(train_dataset=train_dataset, val_dataset=val_dataset, **entity)
            self._eval(train_dataset=train_dataset, val_dataset=val_dataset, **entity)
        else:
            raise ValueError("Value: (train) task name is invalid.")

    def inference(self, infer_dataset: BaseDataset, **entity):
        assert self._infer_result_type in ["probability", "logit"], "Value: infer_type is invalid, get {}".format(self._infer_result_type)
        if self._infer_result_type == "probability":
            return self._predict_prob(infer_dataset, **entity)
        else:
            return self._predict_logit(infer_dataset, **entity)

    def increment(self, increment_dataset: BaseDataset, **entity):
        assert self._model_file_name is not None

        dataset = increment_dataset.get_dataset()
        self._check_bunch(dataset=dataset)
        model_path = os.path.join(self._model_root_path, ConstantValues.model_save)
        model_path = os.path.join(model_path, self._model_file_name)
        if self._task_name == ConstantValues.binary_classification:
            self._binary_increment(train_dataset=increment_dataset,
                                   **entity)
        elif self._task_name == ConstantValues.multiclass_classification:
            self._multiclass_increment(train_dataset=increment_dataset,
                                       **entity)
        elif self._task_name == ConstantValues.regression:
            self._regression_increment(model_path=model_path,
                                       train_dataset=increment_dataset,
                                       **entity)
        else:
            raise ValueError("Value: (increment) task name is invalid.")

    @abc.abstractmethod
    def _binary_train(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        """
        Training model, and this method doesn't contain evaluation.
        :param init_model_path:
        :param train_dataset: BaseDataset object, training dataset.
        :param val_dataset: BaseDataset object, validation dataset.
        :param entity: dict object, including other entity, such as Metric... etc.
        :return:
        """

    @abc.abstractmethod
    def _multiclass_train(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        """
        Training model, and this method doesn't contain evaluation.
        :param init_model_path:
        :param train_dataset: BaseDataset object, training dataset.
        :param val_dataset: BaseDataset object, validation dataset.
        :param entity: dict object, including other entity, such as Metric... etc.
        :return:
        """

    @abc.abstractmethod
    def _regression_train(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        """
        Training model, and this method doesn't contain evaluation.
        :param init_model_path:
        :param train_dataset: BaseDataset object, training dataset.
        :param val_dataset: BaseDataset object, validation dataset.
        :param entity: dict object, including other entity, such as Metric... etc.
        :return:
        """

    @abc.abstractmethod
    def _binary_increment(self, train_dataset: BaseDataset, **entity):
        """
        Training model, and this method doesn't contain evaluation.
        :param init_model_path:
        :param train_dataset: BaseDataset object, training dataset.
        :param val_dataset: BaseDataset object, validation dataset.
        :param entity: dict object, including other entity, such as Metric... etc.
        :return:
        """

    @abc.abstractmethod
    def _multiclass_increment(self, train_dataset: BaseDataset, **entity):
        """
        Training model, and this method doesn't contain evaluation.
        :param init_model_path:
        :param train_dataset: BaseDataset object, training dataset.
        :param val_dataset: BaseDataset object, validation dataset.
        :param entity: dict object, including other entity, such as Metric... etc.
        :return:
        """

    @abc.abstractmethod
    def _regression_increment(self, train_dataset: BaseDataset, **entity):
        """
        Training model, and this method doesn't contain evaluation.
        :param init_model_path:
        :param train_dataset: BaseDataset object, training dataset.
        :param val_dataset: BaseDataset object, validation dataset.
        :param entity: dict object, including other entity, such as Metric... etc.
        :return:
        """

    @abc.abstractmethod
    def _predict_prob(self, infer_dataset: BaseDataset, **entity):
        """
        Predicting probability and get inference result.
        :param infer_dataset: BaseDataset object, evaluating dataset.
        :param entity: dict object, including BaseDataset, Metric... etc.
        :return: None
        """

    @abc.abstractmethod
    def _predict_logit(self, infer_dataset: BaseDataset, **entity):
        """
        Predicting logit and get inference result.
        :param infer_dataset: BaseDataset object, evaluating dataset.
        :param entity: dict object, including BaseDataset, Metric... etc.
        :return: None
        """

    @abc.abstractmethod
    def _eval(self,
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
    def _set_best(self):
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
        self._set_best()

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
