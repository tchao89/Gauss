"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

import os.path

import numpy as np
import pandas as pd
import lightgbm as lgb

from entity.model.multiprocess_model import MultiprocessModelWrapper
from entity.dataset.base_dataset import BaseDataset
from entity.dataset.multiprocess_plain_dataset import MultiprocessPlaintextDataset
from entity.metrics.base_metric import BaseMetric, MetricResult
from utils.bunch import Bunch
from utils.common_component import mkdir, yaml_write, feature_index_generator
from utils.Logger import logger
from utils.base import get_current_memory_gb


class MultiprocessGaussLightgbm(MultiprocessModelWrapper):
    """
    lightgbm running in multiprocess udf mode.
    """
    need_data_clear = False

    def __init__(self, **params):
        super().__init__(
            name=params["name"],
            model_path=params["model_path"],
            model_config_root=params["model_config_root"],
            feature_config_root=params["feature_config_root"],
            task_name=params["task_name"],
            train_flag=params["train_flag"]
        )

        self.model_file_name = self.name + ".txt"
        self.model_config_file_name = self.name + ".yaml"
        self.feature_config_file_name = self.name + ".yaml"

    def __repr__(self):
        pass

    def __load_data(self, dataset: BaseDataset):
        """
        :param dataset: BaseDataset
        :return: lgb.Dataset
        """

        # dataset is a bunch object,
        # including data, target, feature_names, target_names, generated_feature_names.
        if self._train_flag:
            dataset = self._generate_sub_dataset(dataset=dataset)

            lgb_data = lgb.Dataset(
                data=dataset.get("data"),
                label=dataset.get("target"),
                free_raw_data=False,
                silent=True
            )
            logger.info(
                "Method load_data() has finished, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            return lgb_data

        dataset = dataset.get_dataset()
        self._check_bunch(dataset=dataset)
        return dataset.data

    def _initialize_model(self):
        pass

    def train(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        assert self._train_flag is True

        logger.info(
            "Construct lightgbm training dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )
        lgb_train = self.__load_data(dataset=train_dataset)

        assert isinstance(lgb_train, lgb.Dataset)
        logger.info(
            "Construct lightgbm validation dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )
        lgb_eval = self.__load_data(dataset=val_dataset).set_reference(lgb_train)

        logger.info(
            "Set preprocessing parameters for lightgbm, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        if self._model_params is not None:
            self._model_config = {
                "Name": self.name,
                "Normalization": False,
                "Standardization": False,
                "OnehotEncoding": False,
                "ModelParameters": self._model_params
            }

            params = self._model_params

            logger.info(
                "Start training lightgbm model, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            self._model = lgb.train(params,
                                    lgb_train,
                                    num_boost_round=200,
                                    valid_sets=lgb_eval,
                                    early_stopping_rounds=2,
                                    verbose_eval=False)

            logger.info(
                "Training lightgbm model finished, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
        else:
            raise ValueError("Model parameters is None.")

    def predict(self, infer_dataset: BaseDataset, **entity):
        assert self._train_flag is False

        if entity.get("feature_conf") is not None:
            features = feature_index_generator(feature_conf=entity.get("feature_conf"))
            data = infer_dataset.feature_choose(features)

            data_pair = Bunch(data=data, target=None, target_names=None)
            infer_dataset = MultiprocessPlaintextDataset(
                name="inference_data",
                task_name=self._train_flag,
                data_pair=data_pair
            )

        lgb_test = self.__load_data(dataset=infer_dataset)
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

    def eval(self,
             train_dataset: BaseDataset,
             val_dataset: BaseDataset,
             metrics: BaseMetric,
             **entity
             ):

        logger.info(
            "Starting evaluation, with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )
        assert "data" in train_dataset.get_dataset() and "target" in train_dataset.get_dataset()
        assert "data" in val_dataset.get_dataset() and "target" in val_dataset.get_dataset()

        dataset = self._generate_sub_dataset(dataset=train_dataset)
        val_dataset = self._generate_sub_dataset(dataset=val_dataset)

        train_data = dataset.get_dataset().data
        eval_data = val_dataset.get_dataset().data

        train_label = dataset.get_dataset().target
        eval_label = val_dataset.get_dataset().target

        # 默认生成的为预测值的概率值，传入metrics之后再处理.
        logger.info(
            "Starting predicting, with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )
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

        logger.info(
            "train_metric: %s, val_metrics: %s",
            self._train_metrics_result.result,
            self._val_metrics_result.result
        )

    def model_save(self):

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

    def set_weight(self):
        """
        This method can set weight for different label.
        :return: None
        """

    def update_best(self):
        """
        Do not need to operate.
        :return:
        """

    def set_best(self):
        """
        Do not need to operate.
        :return:
        """
