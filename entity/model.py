# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
import abc
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
        super(Entity, self).__init__(
            name=name,
        )

    @abc.abstractmethod
    def train(self, train: BaseDataset, val: BaseDataset = None):
        pass

    @abc.abstractmethod
    def predict(self, test: BaseDataset):
        pass

    @abc.abstractmethod
    def eval(self, test: BaseDataset):
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
