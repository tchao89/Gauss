# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
from __future__ import annotations

import abc

from entity.entity import Entity
from utils.common_component import feature_list_generator


class ModelWrapper(Entity):
    def __init__(self,
                 name: str,
                 model_path: str,
                 model_config_root: str,
                 feature_config_root: str,
                 task_type: str,
                 train_flag: bool
                 ):

        self._model_path = model_path
        self._task_type = task_type
        self._train_flag = train_flag

        # model_config is a dict containing all features and these attributes used in the model.
        self._model_config = None
        # model_config save root
        self._model_config_root = model_config_root
        self._feature_config_root = feature_config_root

        self._model = None

        self._val_metrics_result = None
        self._train_metrics_result = None

        self._model_params = None
        self._feature_conf = None

        self._feature_list = None

        self._best_model = None
        self._best_metrics_result = None
        self._best_model_params = None
        self._best_feature_list = None

        super(ModelWrapper, self).__init__(
            name=name,
        )

    @abc.abstractmethod
    def _initialize_model(self):
        pass

    def update_best_model(self):

        if self._best_model is None:
            self._best_model = self._model

        if self._best_model_params is None:
            self._best_model_params = self._model_params

        if self._best_metrics_result is None:
            self._best_metrics_result = self._val_metrics_result

        if self._best_feature_list is None:
            self._best_feature_list = self._feature_list

        if self._best_metrics_result.__cmp__(self._val_metrics_result) < 0:
            self._best_model = self._model
            self._best_model_params = self._model_params
            self._best_metrics_result = self._val_metrics_result
            self._best_feature_list = self._feature_list

        self.update_best()

    @abc.abstractmethod
    def update_best(self):
        pass

    @abc.abstractmethod
    def train(self, **entity):
        pass

    @abc.abstractmethod
    def predict(self, **entity):
        pass

    @abc.abstractmethod
    def eval(self, **entity):
        pass

    @abc.abstractmethod
    def train_metric(self):
        pass

    @abc.abstractmethod
    def get_train_loss(self):
        pass

    @abc.abstractmethod
    def get_val_loss(self):
        pass

    @property
    def val_metrics(self):
        return self._val_metrics_result

    @property
    def feature_list(self):
        return self._feature_list

    @property
    def model_params(self):
        return self._model_params

    @property
    def model(self):
        return self._model

    @property
    def model_config(self):
        return self._model_config

    @model_config.setter
    def model_config(self, model_config: dict):
        self._model_config = model_config

    @abc.abstractmethod
    def model_save(self):
        pass

    def update_params(self, **params):
        self._model_params.update(params)

    @abc.abstractmethod
    def set_best(self):
        pass

    # This method will convert self.best_object to self.object and set self.best_object to None.
    def set_best_model(self):
        self._model = self._best_model
        self._model_params = self._best_model_params
        self._feature_list = self._best_feature_list
        del self._best_model
        del self._best_model_params
        del self._best_feature_list
        self.set_best()

    def update_feature_conf(self, feature_conf):
        self._feature_conf = feature_conf
        self._feature_list = feature_list_generator(feature_conf=self._feature_conf)
        assert self._feature_list is not None
        return self._feature_list

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
        pass

    @abc.abstractmethod
    def _predict_process(self):
        pass
