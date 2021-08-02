# -*- coding: utf-8 -*-
#
import abc
from typing import List

from entity.model.model import Model


class DNNModel(Model):
    def __init__(self,
                 name: str,
                 model_path: str,
                 task_type: str,
                 train_flag: bool,

                 ):

        super(DNNModel, self).__init__(name=name)

        self._model_path = model_path
        self._task_type = task_type
        self._train_flag = train_flag


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
