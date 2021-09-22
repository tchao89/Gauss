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
import operator

import numpy as np
import pandas as pd
from scipy import special

import core.lightgbm as lgb

from entity.model.single_process_model import SingleProcessModelWrapper
from entity.model.single_process_model import choose_features
from entity.dataset.base_dataset import BaseDataset
from entity.metrics.base_metric import BaseMetric, MetricResult
from entity.losses.base_loss import LossResult

from utils.base import get_current_memory_gb
from utils.base import mkdir
from utils.constant_values import ConstantValues
from utils.yaml_exec import yaml_write
from utils.Logger import logger


class GaussLightgbm(SingleProcessModelWrapper):
    """
    lightgbm object.
    """

    def __init__(self, **params):
        super().__init__(
            name=params["name"],
            model_root_path=params["model_root_path"],
            task_name=params["task_name"],
            train_flag=params["train_flag"]
        )

        self.__model_file_name = self.name + ".txt"
        self.__model_config_file_name = self.name + ".yaml"
        self.__feature_config_file_name = self.name + ".yaml"

        self._loss_function = None
        self._eval_function = None

        self.count = 0

    def __repr__(self):
        pass

    def run(self, **entity):
        if self._train_flag == ConstantValues.train:
            self.train(**entity)
        elif self._train_flag == ConstantValues.inference:
            self.inference(**entity)
        elif self._train_flag == ConstantValues.increment:
            self.increment(**entity)
        else:
            raise ValueError("Value: train flag is invalid.")

    def train(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        dataset = train_dataset.get_dataset()
        self._check_bunch(dataset=dataset)

        if self._task_name == ConstantValues.binary_classification:
            self.binary_train(train_dataset=train_dataset, val_dataset=val_dataset, **entity)
        elif self._task_name == ConstantValues.multiclass_classification:
            self.multiclass_train(train_dataset=train_dataset, val_dataset=val_dataset, **entity)
        elif self._task_name == ConstantValues.regression:
            self.regression_train(train_dataset=train_dataset, val_dataset=val_dataset, **entity)
        else:
            raise ValueError("Value: (train) task name is invalid.")

    def inference(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        dataset = train_dataset.get_dataset()
        self._check_bunch(dataset=dataset)
        if self._task_name == ConstantValues.binary_classification:
            pass
        elif self._task_name == ConstantValues.multiclass_classification:
            pass
        elif self._task_name == ConstantValues.regression:
            pass
        else:
            raise ValueError("Value: (inference) task name is invalid.")

    def increment(self, train_dataset: BaseDataset, val_dataset: BaseDataset, **entity):
        dataset = train_dataset.get_dataset()
        self._check_bunch(dataset=dataset)
        if self._task_name == ConstantValues.binary_classification:
            pass
        elif self._task_name == ConstantValues.multiclass_classification:
            pass
        elif self._task_name == ConstantValues.regression:
            pass
        else:
            raise ValueError("Value: (increment) task name is invalid.")

    @choose_features
    def __load_data(self, **kwargs):
        """
        :param dataset:
        :return: lgb.Dataset
        """
        dataset = kwargs.get("dataset")
        train_flag = kwargs.get("train_flag")
        categorical_list = kwargs.get("categorical_list")
        # dataset is a BaseDataset object, you can use get_dataset() method to get a Bunch object,
        # including data, target, feature_names, target_names, generated_feature_names.
        assert isinstance(dataset.get("data"), pd.DataFrame)

        if train_flag == ConstantValues.train:
            data_shape = dataset.get("data").shape
            label_shape = dataset.get("target").shape
            logger.info("Data shape: {}, label shape: {}".format(data_shape, label_shape))
            assert data_shape[0] == label_shape[0], "Data shape is inconsistent with label shape."

            lgb_data = lgb.Dataset(
                data=dataset.get("data"),
                label=dataset.get("target"),
                categorical_feature=categorical_list,
                free_raw_data=False,
                silent=True
            )

            logger.info(
                "Method load_data() has finished, with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            return lgb_data

        return dataset.get("data")

    def _initialize_model(self):
        pass

    def binary_train(self,
                     train_dataset: BaseDataset,
                     val_dataset: BaseDataset,
                     **entity):
        """
        This method is used to train lightgbm
        model in binary classification.
        :param train_dataset:
        :param val_dataset:
        :param entity:
        :return: None
        """
        assert self._train_flag == ConstantValues.train
        assert self._task_name == ConstantValues.binary_classification

        if entity["loss"] is not None:
            self._loss_function = entity["loss"].loss_fn
            obj_function = self._loss_func
        else:
            obj_function = None

        train_target_names = train_dataset.get_dataset().target_names
        eval_target_names = val_dataset.get_dataset().target_names

        assert operator.eq(train_target_names, eval_target_names), \
            "Value: target_names is different between train_dataset and validation dataset."

        # One label learning is achieved now, multi_label
        # learning will be supported in future.
        self._target_names = list(set(train_target_names).union(set(eval_target_names)))[0]

        train_label_set = pd.unique(train_dataset.get_dataset().target[self._target_names])
        eval_label_set = pd.unique(val_dataset.get_dataset().target[self._target_names])
        train_label_num = len(train_label_set)
        eval_label_num = len(eval_label_set)

        assert train_label_num == eval_label_num and train_label_num == 2 and eval_label_num == 2, \
            "Set of train label is: {}, length: {}, validation label is {}, length is {}, " \
            "and binary classification can not be used.".format(
                train_label_set, train_label_num, eval_label_set, eval_label_num
            )

        if entity["metric"] is not None:
            entity["metric"].label_name = self._target_names
            self._eval_function = entity["metric"].evaluate
            eval_function = self._eval_func
        else:
            eval_function = None

        logger.info(
            "Construct lightgbm training dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        lgb_train = self.__load_data(
            dataset=train_dataset,
            check_bunch=self._check_bunch,
            feature_list=self._feature_list,
            categorical_list=self._categorical_list,
            train_flag=self._train_flag)

        assert isinstance(lgb_train, lgb.Dataset)
        logger.info(
            "Construct lightgbm validation dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        lgb_eval = self.__load_data(
            dataset=val_dataset,
            check_bunch=self._check_bunch,
            feature_list=self._feature_list,
            categorical_list=self._categorical_list,
            train_flag=self._train_flag
        )

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
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
            logger.info(
                "Start training lightgbm model, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            num_boost_round = params.pop("num_boost_round")
            early_stopping_rounds = params.pop("early_stopping_rounds")

            obj_function = None
            eval_function = None

            self._model = lgb.train(
                params=params,
                train_set=lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=lgb_eval,
                categorical_feature=self._categorical_list,
                early_stopping_rounds=early_stopping_rounds,
                fobj=obj_function,
                feval=eval_function,
                verbose_eval=False,
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

    def multiclass_train(self,
                         train_dataset: BaseDataset,
                         val_dataset: BaseDataset,
                         **entity):
        assert self._train_flag == ConstantValues.train

        if entity["loss"] is not None:
            self._loss_function = entity["loss"].loss_fn
            obj_function = self._loss_func
        else:
            obj_function = None

        train_target_names = train_dataset.get_dataset().target_names
        eval_target_names = val_dataset.get_dataset().target_names

        assert operator.eq(train_target_names, eval_target_names), \
            "Value: target_names is different between train_dataset and validation dataset."

        # One label learning is achieved now, multi_label
        # learning will be supported in future.
        self._target_names = list(set(train_target_names).union(set(eval_target_names)))[0]

        train_label_set = pd.unique(train_dataset.get_dataset().target[self._target_names])
        eval_label_set = pd.unique(val_dataset.get_dataset().target[self._target_names])
        train_label_num = len(train_label_set)
        eval_label_num = len(eval_label_set)

        assert train_label_num == eval_label_num and train_label_num > 2 and eval_label_num > 2, \
            "Set of train label is: {}, length: {}, validation label is {}, length is {}, " \
            "and multiclass classification can not be used.".format(
                train_label_set, train_label_num, val_dataset, eval_label_num
            )

        if entity["metric"] is not None:
            entity["metric"].label_name = self._target_names
            self._eval_function = entity["metric"].evaluate
            eval_function = self._eval_func
        else:
            eval_function = None

        logger.info(
            "Construct lightgbm training dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        lgb_train = self.__load_data(
            dataset=train_dataset,
            check_bunch=self._check_bunch,
            feature_list=self._feature_list,
            categorical_list=self._categorical_list,
            train_flag=self._train_flag)

        assert isinstance(lgb_train, lgb.Dataset)
        logger.info(
            "Construct lightgbm validation dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        lgb_eval = self.__load_data(
            dataset=val_dataset,
            check_bunch=self._check_bunch,
            feature_list=self._feature_list,
            categorical_list=self._categorical_list,
            train_flag=self._train_flag
        )

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
            params["objective"] = "multiclass"
            params["metric"] = "multi_logloss"
            params["num_class"] = train_label_num
            logger.info(
                "Training lightgbm model with params: {}".format(params)
            )
            logger.info(
                "Start training lightgbm model, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            num_boost_round = params.pop("num_boost_round")
            early_stopping_rounds = params.pop("early_stopping_rounds")

            obj_function = None
            eval_function = None

            self._model = lgb.train(
                params=params,
                train_set=lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=lgb_eval,
                categorical_feature=self._categorical_list,
                early_stopping_rounds=early_stopping_rounds,
                fobj=obj_function,
                feval=eval_function,
                verbose_eval=False,
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

    def regression_train(self,
                         train_dataset: BaseDataset,
                         val_dataset: BaseDataset,
                         **entity):
        assert self._task_name == ConstantValues.regression

        if entity["loss"] is not None:
            self._loss_function = entity["loss"].loss_fn
            obj_function = self._loss_func
        else:
            obj_function = None

        train_target_names = train_dataset.get_dataset().target_names
        eval_target_names = val_dataset.get_dataset().target_names

        assert operator.eq(train_target_names, eval_target_names), \
            "Value: target_names is different between train_dataset and validation dataset."

        self._target_names = list(set(train_target_names).union(set(eval_target_names)))[0]

        if entity["metric"] is not None:
            entity["metric"].label_name = self._target_names
            self._eval_function = entity["metric"].evaluate
            eval_function = self._eval_func
        else:
            eval_function = None

        logger.info(
            "Construct lightgbm training dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        lgb_train = self.__load_data(
            dataset=train_dataset,
            check_bunch=self._check_bunch,
            feature_list=self._feature_list,
            categorical_list=self._categorical_list,
            train_flag=self._train_flag)

        assert isinstance(lgb_train, lgb.Dataset)
        logger.info(
            "Construct lightgbm validation dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        lgb_eval = self.__load_data(
            dataset=val_dataset,
            check_bunch=self._check_bunch,
            feature_list=self._feature_list,
            categorical_list=self._categorical_list,
            train_flag=self._train_flag
        )

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
            params["objective"] = "multiclass"
            params["metric"] = "multi_logloss"
            logger.info(
                "Start training lightgbm model, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            num_boost_round = params.pop("num_boost_round")
            early_stopping_rounds = params.pop("early_stopping_rounds")
            params["objective"] = "regression"
            params["metric"] = "mse"

            obj_function = None
            eval_function = None

            self._model = lgb.train(
                params=params,
                train_set=lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=lgb_eval,
                categorical_feature=self._categorical_list,
                early_stopping_rounds=early_stopping_rounds,
                fobj=obj_function,
                feval=eval_function,
                verbose_eval=False,
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
        assert self._train_flag == ConstantValues.inference

        lgb_test = self.__load_data(
            dataset=infer_dataset,
            check_bunch=self._check_bunch,
            categorical_list=self._categorical_list,
            feature_list=self._feature_list,
            train_flag=self._train_flag)
        assert os.path.isfile(self._model_save_root + "/" + self.__model_file_name)

        self._model = lgb.Booster(
            model_file=self._model_save_root + "/" + self.__model_file_name
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
             metric: BaseMetric,
             **entity
             ):
        """

        :param train_dataset: BaseDataset object, used to get training metric and loss.
        :param val_dataset: BaseDataset object, used to get validation metric and loss.
        :param metric: BaseMetric object, used to calculate metric.
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

        train_dataset = self._generate_sub_dataset(dataset=train_dataset)
        val_dataset = self._generate_sub_dataset(dataset=val_dataset)

        train_data = train_dataset.get("data")
        eval_data = val_dataset.get("data")

        train_label = train_dataset.get("target")
        eval_label = val_dataset.get("target")

        train_target_names = train_dataset.get("target_names")
        eval_target_names = val_dataset.get("target_names")

        assert operator.eq(train_target_names, eval_target_names), \
            "Value: target_names is different between train_dataset and validation dataset."

        self._target_names = list(set(train_target_names).union(set(eval_target_names)))[0]

        # 默认生成的为预测值的概率值，传入metric之后再处理.
        logger.info(
            "Starting predicting, with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )
        # 默认生成的为预测值的概率值，传入metric之后再处理.
        val_y_pred = self._model.predict(
            eval_data,
            num_iteration=self._model.best_iteration
        )

        train_y_pred = self._model.predict(train_data)

        assert isinstance(val_y_pred, np.ndarray)
        assert isinstance(train_y_pred, np.ndarray)

        train_label = self.__generate_labels_map(
            target=train_label,
            target_names=train_target_names)

        eval_label = self.__generate_labels_map(
            target=eval_label,
            target_names=eval_target_names)

        metric.label_name = self._target_names
        metric.evaluate(predict=val_y_pred, labels_map=eval_label)
        val_metric_result = metric.metric_result

        metric.evaluate(predict=train_y_pred, labels_map=train_label)
        train_metric_result = metric.metric_result

        assert isinstance(val_metric_result, MetricResult)
        assert isinstance(train_metric_result, MetricResult)

        self._val_metric_result = val_metric_result
        self._train_metric_result = train_metric_result

        logger.info("train_metric: %s, val_metric: %s",
                    self._train_metric_result.result,
                    self._val_metric_result.result)

    @classmethod
    def __generate_labels_map(cls, target, target_names):
        assert isinstance(target, pd.DataFrame)
        assert isinstance(target_names, list)

        labels_map = {}
        for feature in target_names:
            labels_map[feature] = target[feature]
        return labels_map

    def model_save(self):
        assert self._model_save_root is not None
        assert self._model is not None

        try:
            assert os.path.isdir(self._model_save_root)

        except AssertionError:
            mkdir(self._model_save_root)

        self._model.save_model(
            os.path.join(
                self._model_save_root,
                self.__model_file_name
            )
        )

        yaml_write(yaml_dict=self._model_config,
                   yaml_file=os.path.join(
                       self._model_config_root,
                       self.__model_config_file_name
                   )
                   )

        yaml_write(yaml_dict={"features": self._feature_list},
                   yaml_file=os.path.join(
                       self._feature_config_root,
                       self.__feature_config_file_name
                   )
                   )

    def set_weight(self, **entity):
        """
        This method can set weight for different label.
        :return: list
        """

    def update_best(self):
        """
        Do not need to operate.
        :return: None
        """

    def set_best(self):
        """
        Do not need to operate.
        :return: None
        """

    def _loss_func(self, preds, train_data):
        assert self._loss_function is not None
        preds = special.expit(preds)
        loss = self._loss_function
        label = train_data.get_label()
        loss_result = loss(score=preds, label=label)

        assert isinstance(loss_result, LossResult)
        return loss_result.grad, loss_result.hess

    def _eval_func(self, preds, train_data):
        assert self._eval_function is not None
        label_map = {self._target_names: train_data.get_label()}
        metric_result = self._eval_function(predict=preds, labels_map=label_map)

        assert isinstance(metric_result, MetricResult)
        assert metric_result.optimize_mode in ["maximize", "minimize"]
        is_higher_better = True if metric_result.optimize_mode == "maximize" else False
        return metric_result.metric_name, float(metric_result.result), is_higher_better
