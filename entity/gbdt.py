# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import lightgbm as lgb

from entity.model import Model
from entity.base_dataset import BaseDataset
from entity.base_metric import BaseMetric
from utils.bunch import Bunch


class GaussXgb(Model):

    def __init__(self, name, model_path, task_type, train_flag):
        super().__init__(name, model_path, task_type, train_flag)
        pass

    def __repr__(self):
        pass

    def predict(self, test: BaseDataset):
        pass

    def eval(self, test: BaseDataset):
        pass

    def get_train_metric(self):
        pass

    def get_train_loss(self):
        pass

    def get_val_loss(self):
        pass

    def update_params(self, **params):
        pass

    def train(self, train: pd.DataFrame, val: pd.DataFrame):
        pass

    def model_save(self):
        pass

    def preprocess(self):
        pass


class GaussLightgbm(Model):
    def __init__(self, **params):
        super(GaussLightgbm, self).__init__(params["name"], params["model_path"], params["task_type"], params["train_flag"])
        self._lgb_model = None
        self._val_metrics = None

    def __repr__(self):
        pass

    def load_data(self, dataset: BaseDataset):
        """

        :param dataset:
        :return:
        """
        # dataset is a bunch object, including data, target, feature_names, target_names, generated_feature_names and
        # val_start.
        val_dataset = dataset.split()
        dataset = dataset.get_dataset()
        self._check_bunch(dataset=dataset)

        # check if this method will create a new project.
        print(dataset.target.shape)
        print(dataset.data.shape)
        print(val_dataset.target.shape)
        print(val_dataset.data.shape)

        train_data = [dataset.data.values, dataset.target.values]
        validation_set = [val_dataset.data.values, val_dataset.target.values]

        lgb_train = lgb.Dataset(data=train_data[0], label=train_data[1])
        lgb_eval = lgb.Dataset(data=validation_set[0], label=validation_set[1], reference=lgb_train)

        return lgb_train, lgb_eval

    @classmethod
    def _check_bunch(cls, dataset: Bunch):
        keys = ["data", "target", "feature_names", "target_names", "generated_feature_names"]
        for key in dataset.keys():
            assert key in keys

    def train(self, dataset: BaseDataset, metrics: BaseMetric, **entity):

        lgb_train, lgb_eval = self.load_data(dataset=dataset)

        if self._model_param_dict is not None:
            params = self._model_param_dict

            self._lgb_model = lgb.train(params,
                                        lgb_train,
                                        num_boost_round=200,
                                        valid_sets=lgb_eval,
                                        early_stopping_rounds=5)
            # predict
            # 默认生成的为预测值的概率值，传入metrics之后再处理.
            y_pred = self._lgb_model.predict(lgb_eval.get_data(), num_iteration=self._lgb_model.best_iteration)

            metrics.evaluate(predict=y_pred, labels_map=dataset.get_dataset().target)
            metrics = metrics.metrics_result()

        else:
            raise ValueError("Model parameters is None.")
        return metrics

    def predict(self, test: BaseDataset):
        pass

    def preprocess(self):
        pass

    def eval(self, test: BaseDataset):
        pass

    def get_train_loss(self):
        pass

    def get_val_loss(self):
        pass

    def get_train_metric(self):
        pass

    @property
    def val_metrics(self):
        return self._val_metrics

    def model_save(self):
        pass

    def update_params(self, **params):
        self._model_param_dict.update(params)

    def set_weight(self):
        pass


class GaussCatBoost(Model):

    def __init__(self, name: str, model_path: str, task_type: str, train_flag: bool):
        super().__init__(name, model_path, task_type, train_flag)
        pass

    def __repr__(self):
        pass

    def train(self, train: pd.DataFrame, val: pd.DataFrame):
        pass

    def predict(self, test: BaseDataset):
        pass

    def eval(self, test: BaseDataset):
        pass

    def get_train_metric(self):
        pass

    def get_train_loss(self):
        pass

    def get_val_loss(self):
        pass

    def update_params(self, **params):
        pass

    def preprocess(self):
        pass
