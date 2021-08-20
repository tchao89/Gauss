# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

import os.path

import numpy as np
import pandas as pd
import lightgbm as lgb

from entity.model.model import ModelWrapper
from entity.dataset.base_dataset import BaseDataset
from entity.dataset.multiprocess_plain_dataset import PlaintextDataset
from entity.metrics.base_metric import BaseMetric, MetricResult
from utils.bunch import Bunch
from utils.common_component import mkdir, yaml_write, feature_index_generator
from utils.Logger import logger


class MultiprocessGaussLightgbm(ModelWrapper):
    def __init__(self, **params):
        super(MultiprocessGaussLightgbm, self).__init__(name=params["name"],
                                                        model_path=params["model_path"],
                                                        model_config_root=params["model_config_root"],
                                                        feature_config_root=params["feature_config_root"],
                                                        task_type=params["task_type"],
                                                        train_flag=params["train_flag"])
        self.need_data_clear = False

        self.model_file_name = self.name + ".txt"
        self.model_config_file_name = self.name + ".yaml"
        self.feature_config_file_name = self.name + ".yaml"

    def __repr__(self):
        pass

    def load_data(self, dataset: BaseDataset):
        """
        :param dataset:
        :return: lgb.Dataset
        """

        # dataset is a bunch object, including data, target, feature_names, target_names, generated_feature_names.
        if self._train_flag:

            if self._feature_list is not None:
                data = dataset.feature_choose(self._feature_list)
                target = dataset.get_dataset().target
                data_pair = Bunch(data=data, target=target, target_names=dataset.get_dataset().target_names)
                dataset = PlaintextDataset(name="train_data", task_type=self._task_type, data_pair=data_pair)

            dataset = dataset.get_dataset()
            self._check_bunch(dataset=dataset)
            train_data = [dataset.data, dataset.target.flatten()]
            lgb_data = lgb.Dataset(data=train_data[0], label=train_data[1], free_raw_data=False, silent=True)

            return lgb_data
        else:
            dataset = dataset.get_dataset()
            self._check_bunch(dataset=dataset)
            return dataset.data

    def _initialize_model(self):
        pass

    @classmethod
    def _check_bunch(cls, dataset: Bunch):
        keys = ["data", "target", "feature_names", "target_names", "generated_feature_names"]
        for key in dataset.keys():
            assert key in keys

    def update_feature_conf(self, feature_conf):
        self._feature_conf = feature_conf
        self._feature_list = feature_index_generator(feature_conf=self._feature_conf)
        assert self._feature_list is not None
        return self._feature_list

    def train(self, dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        assert self._train_flag is True

        lgb_train = self.load_data(dataset=dataset)
        lgb_eval = self.load_data(dataset=val_dataset)

        if self._model_params is not None:

            self._model_config = {
                "Name": self.name,
                "Normalization": False,
                "Standardization": False,
                "OnehotEncoding": False,
                "ModelParameters": self._model_params
            }

            params = self._model_params
            self._model = lgb.train(params,
                                    lgb_train,
                                    num_boost_round=200,
                                    valid_sets=lgb_eval,
                                    early_stopping_rounds=2,
                                    verbose_eval=False)

        else:
            raise ValueError("Model parameters is None.")

    def predict(self, dataset: BaseDataset, **entity):
        assert self._train_flag is False

        if entity.get("feature_conf") is not None:
            features = feature_index_generator(feature_conf=entity.get("feature_conf"))
            data = dataset.feature_choose(features)

            data_pair = Bunch(data=data, target=None, target_names=None)
            dataset = PlaintextDataset(name="inference_data", task_type=self._train_flag, data_pair=data_pair)

        lgb_test = self.load_data(dataset=dataset)
        assert os.path.isfile(self._model_path + "/" + self.model_file_name)

        self._model = lgb.Booster(model_file=self._model_path + "/" + self.model_file_name)

        inference_result = self._model.predict(lgb_test)
        inference_result = pd.DataFrame({"result": inference_result})
        return inference_result

    def preprocess(self):
        pass

    def _train_preprocess(self):
        pass

    def _predict_process(self):
        pass

    def eval(self, dataset: BaseDataset, val_dataset: BaseDataset, metrics: BaseMetric, **entity):
        lgb_eval = self.load_data(dataset=val_dataset).construct()
        lgb_train = self.load_data(dataset=dataset).construct()
        # 默认生成的为预测值的概率值，传入metrics之后再处理.

        val_y_pred = self._model.predict(lgb_eval.get_data(), num_iteration=self._model.best_iteration)
        train_y_pred = self._model.predict(lgb_train.get_data())

        assert isinstance(val_y_pred, np.ndarray)
        assert isinstance(train_y_pred, np.ndarray)
        assert isinstance(lgb_eval.get_label(), np.ndarray)
        assert isinstance(lgb_train.get_label(), np.ndarray)

        metrics.evaluate(predict=val_y_pred, labels_map=lgb_eval.get_label())
        val_metrics_result = metrics.metrics_result

        metrics.evaluate(predict=train_y_pred, labels_map=lgb_train.get_label())
        train_metrics_result = metrics.metrics_result

        assert isinstance(val_metrics_result, MetricResult)
        assert isinstance(train_metrics_result, MetricResult)

        self._val_metrics_result = val_metrics_result
        self._train_metrics_result = train_metrics_result
        logger.info("train_metric: " + str(self._train_metrics_result.result) + "   val_metrics: " + str(
            self._val_metrics_result.result))

    def get_train_loss(self):
        pass

    def get_val_loss(self):
        pass

    @property
    def train_metric(self):
        return self._train_metrics_result

    @property
    def val_metrics(self):
        return self._val_metrics_result

    def model_save(self, model_path=None):

        if model_path is not None:
            self._model_path = model_path

        assert self._model_path is not None
        assert self._model is not None

        try:
            assert os.path.isdir(self._model_path)

        except AssertionError:
            mkdir(self._model_path)

        self._model.save_model(os.path.join(self._model_path, self.model_file_name))

        yaml_write(yaml_dict=self._model_config,
                   yaml_file=os.path.join(self._model_config_root, self.model_config_file_name))

        assert self._feature_list is not None
        yaml_write(yaml_dict={"features": self._feature_list},
                   yaml_file=os.path.join(self._feature_config_root, self.feature_config_file_name))

    def update_params(self, **params):
        if self._model_params is None:
            self._model_params = {}

        self._model_params.update(params)

    def set_weight(self):
        pass

    def update_best(self):
        pass

    def set_best(self):
        pass
