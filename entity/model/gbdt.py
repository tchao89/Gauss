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
from utils.base import get_current_memory_gb
from utils.bunch import Bunch
from utils.common_component import mkdir, yaml_write, feature_list_generator
from utils.Logger import logger


class GaussLightgbm(ModelWrapper):
    need_data_clear = False

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

    def __repr__(self):
        pass

    def generate_sub_dataset(self, dataset: BaseDataset):
        if self._feature_list is not None:
            data = dataset.feature_choose(self._feature_list)
            target = dataset.get_dataset().target
            target_names = dataset.get_dataset().target_names
            data_pair = Bunch(data=data, target=target, target_names=target_names)
            dataset = PlaintextDataset(name="train_data", task_type=self._task_type, data_pair=data_pair)

        return dataset

    def load_data(self, dataset: BaseDataset):
        """
        :param dataset:
        :return: lgb.Dataset
        """

        # dataset is a bunch object, including data, target, feature_names, target_names, generated_feature_names.
        if self._train_flag:
            dataset = self.generate_sub_dataset(dataset=dataset)

            logger.info("Reading base dataset, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            dataset = dataset.get_dataset()

            logger.info("Check base dataset, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            self._check_bunch(dataset=dataset)

            logger.info("Construct lgb.Dataset object in load_data method, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            train_data = [dataset.data.values, dataset.target.values.flatten()]

            lgb_data = lgb.Dataset(data=train_data[0], label=train_data[1], free_raw_data=False, silent=True)
            logger.info("Method load_data() has finished, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
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

    def train(self, dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        assert self._train_flag is True

        logger.info("Construct lightgbm training dataset, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        lgb_train = self.load_data(dataset=dataset)

        assert isinstance(lgb_train, lgb.Dataset)
        logger.info("Construct lightgbm validation dataset, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        lgb_eval = self.load_data(dataset=val_dataset).set_reference(lgb_train)

        logger.info("Set preprocessing parameters for lightgbm, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])

        if self._model_params is not None:

            self._model_config = {
                "Name": self.name,
                "Normalization": False,
                "Standardization": False,
                "OnehotEncoding": False,
                "ModelParameters": self._model_params
            }

            params = self._model_params
            logger.info("Start training lightgbm model, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            self._model = lgb.train(params,
                                    lgb_train,
                                    num_boost_round=200,
                                    valid_sets=lgb_eval,
                                    early_stopping_rounds=2,
                                    verbose_eval=False)

            logger.info("Training lightgbm model finished, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])

        else:
            raise ValueError("Model parameters is None.")

    def predict(self, dataset: BaseDataset, **entity):
        assert self._train_flag is False

        if entity.get("feature_conf") is not None:
            features = feature_list_generator(feature_conf=entity.get("feature_conf"))
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
        logger.info("Starting evaluation, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        assert "data" in dataset.get_dataset() and "target" in dataset.get_dataset()
        assert "data" in val_dataset.get_dataset() and "target" in val_dataset.get_dataset()
        dataset = self.generate_sub_dataset(dataset=dataset)
        val_dataset = self.generate_sub_dataset(dataset=val_dataset)

        train_data = dataset.get_dataset().data.values
        eval_data = val_dataset.get_dataset().data.values

        train_label = dataset.get_dataset().target.values
        eval_label = val_dataset.get_dataset().target.values

        # 默认生成的为预测值的概率值，传入metrics之后再处理.
        logger.info("Starting predicting, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        # 默认生成的为预测值的概率值，传入metrics之后再处理.
        val_y_pred = self._model.predict(eval_data, num_iteration=self._model.best_iteration)
        train_y_pred = self._model.predict(train_data)

        assert isinstance(val_y_pred, np.ndarray)
        assert isinstance(train_y_pred, np.ndarray)
        assert isinstance(eval_label, np.ndarray)
        assert isinstance(train_label, np.ndarray)

        metrics.evaluate(predict=val_y_pred, labels_map=eval_label)
        val_metrics_result = metrics.metrics_result

        metrics.evaluate(predict=train_y_pred, labels_map=train_label)
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
