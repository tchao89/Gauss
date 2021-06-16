# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
import abc

import pandas as pd

from entity.base_dataset import BaseDataset
from entity.entity import Entity


class Model(Entity):
    def __init__(self,
                 name: str,
                 model_path: str,
                 task_type: str,
                 train_flag: bool
                 ):

        self._model_path = model_path
        self._task_type = task_type
        self._train_flag = train_flag
        self._train_finished = False
        self._model_param_dict = {}

        super(Entity, self).__init__(
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
