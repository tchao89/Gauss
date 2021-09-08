"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
GBDT model instances, containing lightgbm, xgboost and catboost.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

import os.path

import numpy as np
import pandas as pd
import lightgbm as lgb

from entity.model.single_process_model import SingleProcessModelWrapper
from entity.dataset.base_dataset import BaseDataset
from entity.metrics.base_metric import BaseMetric, MetricResult
from utils.base import get_current_memory_gb
from utils.common_component import mkdir, yaml_write
from utils.Logger import logger


class GaussLightgbm(SingleProcessModelWrapper):
    """
    lightgbm object.
    """
    # This flag is bool values, if true,
    # dataset must use data clear component when training this model.
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

        self.__model_file_name = self.name + ".txt"
        self.__model_config_file_name = self.name + ".yaml"
        self.__feature_config_file_name = self.name + ".yaml"

        self.count = 0

    def __repr__(self):
        pass

    def __load_data(self, dataset: BaseDataset):
        """
        :param dataset: BaseDataset
        :return: lgb.Dataset
        """

        # dataset is a BaseDataset object, you can use get_dataset() method to get a Bunch object,
        # including data, target, feature_names, target_names, generated_feature_names.
        dataset = self._generate_sub_dataset(dataset=dataset)

        if self._train_flag:
            lgb_data = lgb.Dataset(
                data=dataset.get("data"),
                label=dataset.get("target"),
                free_raw_data=False,
                silent=True
            )

            logger.info(
                "Method load_data() has finished, with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            return lgb_data

        self._check_bunch(dataset=dataset)
        return dataset.get("data")

    def _initialize_model(self):
        pass

    def train(self,
              train_dataset: BaseDataset,
              val_dataset: BaseDataset,
              **entity
              ):
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
            self._model = lgb.train(
                params=params,
                train_set=lgb_train,
                num_boost_round=200,
                valid_sets=lgb_eval,
                early_stopping_rounds=2,
                verbose_eval=False
            )

            logger.info(
                "Training lightgbm model finished, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

        else:
            raise ValueError("Model parameters is None.")
        self.count += 1

    def predict(self,
                infer_dataset: BaseDataset,
                **entity
                ):
        assert self._train_flag is False

        lgb_test = self.__load_data(dataset=infer_dataset)
        assert os.path.isfile(self._model_path + "/" + self.__model_file_name)

        self._model = lgb.Booster(
            model_file=self._model_path + "/" + self.__model_file_name
        )

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
        """

        :param train_dataset: BaseDataset object, used to get training metric and loss.
        :param val_dataset: BaseDataset object, used to get validation metric and loss.
        :param metrics: BaseMetric object, used to calculate metrics.
        :param entity: dict object, including other entity.
        :return: None
        """
        logger.info(
            "Starting evaluation, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        assert "data" in train_dataset.get_dataset() and "target" in train_dataset.get_dataset()
        assert "data" in val_dataset.get_dataset() and "target" in val_dataset.get_dataset()

        dataset = self._generate_sub_dataset(dataset=train_dataset)
        val_dataset = self._generate_sub_dataset(dataset=val_dataset)

        train_data = dataset.get("data")
        eval_data = val_dataset.get("data")

        train_label = dataset.get("target")
        eval_label = val_dataset.get("target")

        # 默认生成的为预测值的概率值，传入metrics之后再处理.
        logger.info(
            "Starting predicting, with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )
        # 默认生成的为预测值的概率值，传入metrics之后再处理.
        val_y_pred = self._model.predict(
            eval_data,
            num_iteration=self._model.best_iteration
        )

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

        logger.info("train_metric: %s, val_metrics: %s",
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

        self._model.save_model(
            os.path.join(
                self._model_path,
                self.__model_file_name
            )
        )

        yaml_write(yaml_dict=self._model_config,
                   yaml_file=os.path.join(
                       self._model_config_root,
                       self.__model_config_file_name
                        )
                   )

        assert self._feature_list is not None
        yaml_write(yaml_dict={"features": self._feature_list},
                   yaml_file=os.path.join(
                       self._feature_config_root,
                       self.__feature_config_file_name
                   )
                   )

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
