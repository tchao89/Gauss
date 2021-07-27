# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
from __future__ import annotations

import abc

from entity.entity import Entity
from utils.common_component import feature_list_generator


class Model(Entity):
    def __init__(self,
                 name: str,
                 model_path: str,
                 model_config_root: str,
                 task_type: str,
                 train_flag: bool,
                 model_config: dict = None
                 ):

        self._model_path = model_path
        self._task_type = task_type
        self._train_flag = train_flag
        self._train_finished = False

        # model_config is a dict containing all features and these attributes used in the model.
        self._model_config = model_config
        # model_config save root
        self._model_config_root = model_config_root

        self._model_param_dict = {}
        self._feature_list = feature_list_generator(feature_dict=self._model_config)

        super(Model, self).__init__(
            name=name,
        )

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
    def get_train_metric(self):
        pass

    @abc.abstractmethod
    def get_train_loss(self):
        pass

    @abc.abstractmethod
    def get_val_loss(self):
        pass

    def update_params(self, **params):
        self._model_param_dict.update(params)

    @abc.abstractmethod
    def preprocess(self):
        """
        This method is used to implement Normalization, Standardization, which need self._train_flag parameters.
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
