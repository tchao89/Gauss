# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

import os.path

import numpy
import lightgbm as lgb

from entity.model.model import Model
from entity.dataset.base_dataset import BaseDataset
from entity.metrics.base_metric import BaseMetric, MetricResult
from utils.bunch import Bunch


class GaussLightgbm(Model):
    def __init__(self, **params):
        super(GaussLightgbm, self).__init__(params["name"], params["model_path"], params["task_type"],
                                            params["train_flag"])
        self.file_name = self.name + ".txt"

        self._lgb_model = None
        self._val_metrics = None

        # lgb.Dataset
        self.lgb_train = None
        self.lgb_eval = None
        self.lgb_test = None

    def __repr__(self):
        pass

    def load_data(self, dataset: BaseDataset, val_dataset: BaseDataset = None):
        """

        :param val_dataset:
        :param dataset:
        :return: lgb.Dataset
        """
        # dataset is a bunch object, including data, target, feature_names, target_names, generated_feature_names.
        if self._train_flag:
            assert val_dataset is not None
            dataset = dataset.get_dataset()
            val_dataset = val_dataset.get_dataset()

            self._check_bunch(dataset=dataset)
            self._check_bunch(dataset=val_dataset)

            train_data = [dataset.data.values, dataset.target.values]
            validation_set = [val_dataset.data.values, val_dataset.target.values]

            lgb_train = lgb.Dataset(data=train_data[0], label=train_data[1], free_raw_data=False)
            lgb_eval = lgb.Dataset(data=validation_set[0], label=validation_set[1], reference=lgb_train,
                                   free_raw_data=False)

            return lgb_train, lgb_eval
        else:
            assert val_dataset is None

            dataset = dataset.get_dataset()
            self._check_bunch(dataset=dataset)
            return dataset.data.values

    @classmethod
    def _check_bunch(cls, dataset: Bunch):
        keys = ["data", "target", "feature_names", "target_names", "generated_feature_names"]
        for key in dataset.keys():
            assert key in keys

    def train(self, dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        assert self._train_flag is True

        self.lgb_train, self.lgb_eval = self.load_data(dataset=dataset, val_dataset=val_dataset)

        if self._model_param_dict is not None:
            params = self._model_param_dict
            self._lgb_model = lgb.train(params,
                                        self.lgb_train,
                                        num_boost_round=200,
                                        valid_sets=self.lgb_eval,
                                        early_stopping_rounds=2,
                                        verbose_eval=0)

        else:
            raise ValueError("Model parameters is None.")

    def predict(self, test_dataset: BaseDataset):
        assert self._train_flag is False

        self.lgb_test = self.load_data(dataset=test_dataset)

        assert os.path.isfile(self._model_path + self.file_name)
        self._lgb_model = lgb.Booster(model_file=self._model_path + self.file_name)

        inference_result = self._lgb_model.predict(self.lgb_test)
        return inference_result

    def preprocess(self):
        pass

    def eval(self, metrics: BaseMetric, **entity):
        # 默认生成的为预测值的概率值，传入metrics之后再处理.
        y_pred = self._lgb_model.predict(self.lgb_eval.get_data(), num_iteration=self._lgb_model.best_iteration)

        assert isinstance(y_pred, numpy.ndarray)
        assert isinstance(self.lgb_eval.get_label(), numpy.ndarray)

        metrics.evaluate(predict=y_pred, labels_map=self.lgb_eval.get_label())
        metrics = metrics.metrics_result
        assert isinstance(metrics, MetricResult)
        self._val_metrics = metrics

        return metrics

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
        assert self._lgb_model is not None

        self._lgb_model.save_model(self._model_path + self.file_name)

    def update_params(self, **params):
        self._model_param_dict.update(params)

    def set_weight(self):
        pass

    @property
    def need_data_clear(self):
        assert isinstance(self._need_data_clear, bool)
        return self._need_data_clear
