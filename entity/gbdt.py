# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from entity.model import Model
from entity.base_dataset import BaseDataset
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


class GaussLightgbm(Model):
    def __init__(self, name: str, model_path, task_name, train_flag, metric_name: str):
        super(GaussLightgbm, self).__init__(name, model_path, task_name, train_flag)
        self.metric_name = metric_name
        self.lgb_model = None

    def __repr__(self):
        pass

    def load_data(self, **entity):
        """
        this method's parameters is train_data and validation_set.
        :param entity:
        :return:
        """
        # dataset is a bunch object, including data, target, feature_names, target_names, generated_feature_names and
        # val_start.
        dataset = entity["dataset"].get_data()
        self._check_bunch(dataset=dataset)

        val_start = dataset.val_start
        assert isinstance(val_start, int) and dataset.data.shape[0] > val_start > 0

        # check if this method will create a new project.
        train_data = [dataset.data[:val_start], dataset.target[:val_start]]
        validation_set = [dataset.data[val_start:], dataset.target[val_start:]]

        lgb_train = lgb.Dataset(data=train_data[0], label=train_data[1])
        lgb_eval = lgb.Dataset(data=validation_set[0], label=validation_set[1], reference=lgb_train)

        return lgb_train, lgb_eval

    @classmethod
    def _check_bunch(cls, dataset: Bunch):
        keys = ["data", "target", "feature_names", "target_names", "generated_feature_names", "val_start"]
        for key in dataset.keys():
            assert key in keys

    def train(self, **entity):
        assert "dataset" in entity

        lgb_train, lgb_eval = self.load_data(**entity)
        if self._model_param_dict is not None:
            params = self._model_param_dict

            self.lgb_model = lgb.train(params,
                                       lgb_train,
                                       num_boost_round=200,
                                       valid_sets=lgb_eval,
                                       early_stopping_rounds=5)
            # predict
            # 默认生成的为预测值的概率值，传入metrics之后再处理.
            y_pred = self.lgb_model.predict(lgb_eval.get_data(), num_iteration=self.lgb_model.best_iteration)

            # just for test, metrics object will complete soon.
            rmse = roc_auc_score(y_true=lgb_eval.get_label(), y_score=y_pred)
            print('The rmse of prediction is:', rmse)

        else:
            raise ValueError("Model parameters is None.")

    def predict(self, test: BaseDataset):
        pass

    def eval(self, test: BaseDataset):
        pass

    def get_train_loss(self):
        pass

    def get_val_loss(self):
        pass

    def get_train_metric(self):
        pass

    def get_val_metrics(self):
        pass

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
