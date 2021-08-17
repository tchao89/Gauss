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
from entity.dataset.plain_dataset import PlaintextDataset
from entity.metrics.base_metric import BaseMetric, MetricResult
from utils.bunch import Bunch
from utils.common_component import mkdir, yaml_write, feature_list_generator
from utils.Logger import logger


class GaussLightgbm(ModelWrapper):
    def __init__(self, **params):
        super(GaussLightgbm, self).__init__(name=params["name"],
                                            model_path=params["model_path"],
                                            model_config_root=params["model_config_root"],
                                            feature_config_root=params["feature_config_root"],
                                            task_type=params["task_type"],
                                            train_flag=params["train_flag"])
        self.need_data_clear = False

        self.model_file_name = self.name + ".txt"
        self.model_config_file_name = self.name + ".yaml"
        self.feature_config_file_name = self.name + ".yaml"

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

            if self._feature_list is not None:
                data = dataset.feature_choose(self._feature_list)
                target = dataset.get_dataset().target
                data_pair = Bunch(data=data, target=target, target_names=dataset.get_dataset().target_names)
                dataset = PlaintextDataset(name="train_data", task_type=self._task_type, data_pair=data_pair)

                val_data = val_dataset.feature_choose(self._feature_list)
                val_target = val_dataset.get_dataset().target
                val_data_pair = Bunch(data=val_data, target=val_target,
                                      target_names=dataset.get_dataset().target_names)
                val_dataset = PlaintextDataset(name="val_dataset", task_type=self._task_type, data_pair=val_data_pair)

            assert val_dataset is not None
            dataset = dataset.get_dataset()
            val_dataset = val_dataset.get_dataset()

            self._check_bunch(dataset=dataset)
            self._check_bunch(dataset=val_dataset)

            train_data = [dataset.data.values, dataset.target.values.flatten()]
            validation_set = [val_dataset.data.values, val_dataset.target.values.flatten()]

            lgb_train = lgb.Dataset(data=train_data[0], label=train_data[1], free_raw_data=False, silent=True)
            lgb_eval = lgb.Dataset(data=validation_set[0], label=validation_set[1], reference=lgb_train,
                                   free_raw_data=False, silent=True)

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
                                    self.lgb_train,
                                    num_boost_round=200,
                                    valid_sets=self.lgb_eval,
                                    early_stopping_rounds=2,
                                    verbose_eval=False)

        else:
            raise ValueError("Model parameters is None.")

    def predict(self, dataset: BaseDataset, **entity):
        assert self._train_flag is False

        if entity.get("feature_conf") is not None:
            features = feature_list_generator(feature_conf=entity.get("feature_conf"))
            data = dataset.feature_choose(features)

            data_pair = Bunch(data=data, target=None, target_names=None)
            dataset = PlaintextDataset(name="inference_data", task_type=self._train_flag, data_pair=data_pair)

        self.lgb_test = self.load_data(dataset=dataset)
        assert os.path.isfile(self._model_path + "/" + self.model_file_name)

        self._model = lgb.Booster(model_file=self._model_path + "/" + self.model_file_name)

        inference_result = self._model.predict(self.lgb_test)
        inference_result = pd.DataFrame({"result": inference_result})
        return inference_result

    def preprocess(self):
        pass

    def _train_preprocess(self):
        pass

    def _predict_process(self):
        pass

    def eval(self, metrics: BaseMetric, **entity):
        # 默认生成的为预测值的概率值，传入metrics之后再处理.
        val_y_pred = self._model.predict(self.lgb_eval.get_data(), num_iteration=self._model.best_iteration)
        train_y_pred = self._model.predict(self.lgb_train.get_data())

        assert isinstance(val_y_pred, np.ndarray)
        assert isinstance(train_y_pred, np.ndarray)
        assert isinstance(self.lgb_eval.get_label(), np.ndarray)
        assert isinstance(self.lgb_train.get_label(), np.ndarray)

        metrics.evaluate(predict=val_y_pred, labels_map=self.lgb_eval.get_label())
        val_metrics_result = metrics.metrics_result

        metrics.evaluate(predict=train_y_pred, labels_map=self.lgb_train.get_label())
        train_metrics_result = metrics.metrics_result

        assert isinstance(val_metrics_result, MetricResult)
        assert isinstance(train_metrics_result, MetricResult)

        self._val_metrics_result = val_metrics_result
        self._train_metrics_result = train_metrics_result
        logger.info("train_metric: " + str(self._train_metrics_result.result) + "   val_metrics: " + str(self._val_metrics_result.result))

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
