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

from entity.model.model import ModelWrapper
from entity.model.package_dataset import PackageDataset
from entity.dataset.base_dataset import BaseDataset
from entity.metrics.base_metric import BaseMetric, MetricResult
from entity.losses.base_loss import LossResult

from utils.base import get_current_memory_gb
from utils.base import mkdir
from utils.bunch import Bunch
from utils.constant_values import ConstantValues
from utils.yaml_exec import yaml_write
from utils.Logger import logger


class GaussLightgbm(ModelWrapper):
    """
    lightgbm object.
    """

    def __init__(self, **params):
        assert params[ConstantValues.train_flag] in [ConstantValues.train,
                                                     ConstantValues.increment,
                                                     ConstantValues.inference]
        if params[ConstantValues.train_flag] == ConstantValues.train:
            super().__init__(
                name=params[ConstantValues.name],
                model_root_path=params[ConstantValues.model_root_path],
                init_model_root=params[ConstantValues.init_model_root],
                task_name=params[ConstantValues.task_name],
                train_flag=params[ConstantValues.train_flag],
                metric_eval_used_flag=params[ConstantValues.metric_eval_used_flag]
            )
        elif params[ConstantValues.train_flag] == ConstantValues.increment:
            super().__init__(
                name=params[ConstantValues.name],
                model_root_path=params[ConstantValues.model_root_path],
                init_model_root=params[ConstantValues.init_model_root],
                task_name=params[ConstantValues.task_name],
                train_flag=params[ConstantValues.train_flag],
                decay_rate=params[ConstantValues.decay_rate],
                metric_eval_used_flag=params[ConstantValues.metric_eval_used_flag]
            )
        else:
            assert params[ConstantValues.train_flag] == ConstantValues.inference
            super().__init__(
                name=params[ConstantValues.name],
                model_root_path=params[ConstantValues.model_root_path],
                init_model_root=params[ConstantValues.init_model_root],
                increment_flag=params[ConstantValues.increment_flag],
                infer_result_type=params[ConstantValues.infer_result_type],
                task_name=params[ConstantValues.task_name],
                train_flag=params[ConstantValues.train_flag],
                metric_eval_used_flag=params[ConstantValues.metric_eval_used_flag]
            )

        self._model_file_name = self._name + ".txt"
        self._model_config_file_name = self._name + ".yaml"
        self._feature_config_file_name = self._name + ".yaml"

        self._loss_function = None
        self._eval_function = None

        self.count = 0

    def __repr__(self):
        pass

    @PackageDataset
    def __load_data(self, **kwargs):
        """
        :param dataset:
        :return: lgb.Dataset
        """
        dataset_bunch = kwargs.get("dataset")
        train_flag = kwargs.get("train_flag")

        categorical_list = dataset_bunch.categorical_list
        # dataset is a BaseDataset object, you can use get_dataset() method to get a Bunch object,
        # including data, target, feature_names, target_names, generated_feature_names.
        assert isinstance(dataset_bunch.data, pd.DataFrame)
        if train_flag == ConstantValues.train or train_flag == ConstantValues.increment:
            target_names = dataset_bunch.target_names[0]
            data_shape = dataset_bunch.data.shape
            label_shape = dataset_bunch.target.shape
            logger.info("Data shape: {}, label shape: {}".format(data_shape, label_shape))
            assert data_shape[0] == label_shape[0], "Data shape is inconsistent with label shape."

            if dataset_bunch.dataset_weight is not None:
                weight = dataset_bunch.dataset_weight
            else:
                weight = None

            if isinstance(weight, pd.DataFrame):
                weight = weight.values.flatten()

            lgb_data = lgb.Dataset(
                data=dataset_bunch.data,
                label=dataset_bunch.target,
                categorical_feature=categorical_list,
                weight=weight,
                free_raw_data=False,
                silent=True
            )

            logger.info(
                "Method load_data() has finished, with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            return lgb_data, dataset_bunch
        return dataset_bunch

    def _initialize_model(self):
        pass

    def _binary_train(self,
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
        init_model_path = self._init_model_root
        if init_model_path:
            assert os.path.isfile(init_model_path), \
                "Value: init_model_path({}) is not a valid model path.".format(
                    init_model_path)

        params = self._model_params
        params["objective"] = "binary"

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

        if self._metric_eval_used_flag and entity["metric"] is not None:
            entity["metric"].label_name = self._target_names
            self._eval_function = entity["metric"].evaluate
            eval_function = self._eval_func
        else:
            params["metric"] = "binary_logloss"
            eval_function = None

        logger.info(
            "Construct lightgbm training dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        lgb_entity = self.__lgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        lgb_train = lgb_entity.lgb_train
        lgb_eval = lgb_entity.lgb_eval

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

            logger.info(
                "Start training lightgbm model, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            num_boost_round = params.pop("num_boost_round")
            early_stopping_rounds = params.pop("early_stopping_rounds")
            assert isinstance(lgb_train, lgb.Dataset)

            self._model = lgb.train(
                params=params,
                train_set=lgb_train,
                init_model=init_model_path,
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

    def _multiclass_train(self,
                          train_dataset: BaseDataset,
                          val_dataset: BaseDataset,
                          **entity):
        assert self._train_flag == ConstantValues.train
        init_model_path = self._init_model_root

        params = self._model_params
        params["objective"] = "multiclass"

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

        params["num_class"] = train_label_num

        assert train_label_num == eval_label_num and train_label_num > 2 and eval_label_num > 2, \
            "Set of train label is: {}, length: {}, validation label is {}, length is {}, " \
            "and multiclass classification can not be used.".format(
                train_label_set, train_label_num, val_dataset, eval_label_num
            )

        if self._metric_eval_used_flag and entity["metric"] is not None:
            entity["metric"].label_name = self._target_names
            self._eval_function = entity["metric"].evaluate
            eval_function = self._eval_func
        else:
            params["metric"] = "multi_logloss"
            eval_function = None

        logger.info(
            "Construct lightgbm training dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        lgb_entity = self.__lgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        lgb_train = lgb_entity.lgb_train
        lgb_eval = lgb_entity.lgb_eval

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

            self._model = lgb.train(
                params=params,
                train_set=lgb_train,
                init_model=init_model_path,
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

    def _regression_train(self,
                          train_dataset: BaseDataset,
                          val_dataset: BaseDataset,
                          **entity):
        assert self._task_name == ConstantValues.regression
        init_model_path = self._init_model_root

        params = self._model_params
        params["objective"] = "regression"

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
        entity["metric"].label_name = self._target_names

        if self._metric_eval_used_flag and entity["metric"] is not None:
            self._eval_function = entity["metric"].evaluate
            eval_function = self._eval_func
        else:
            params["metric"] = "mse"
            eval_function = None

        logger.info(
            "Construct lightgbm training dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        lgb_entity = self.__lgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        lgb_train = lgb_entity.lgb_train
        lgb_eval = lgb_entity.lgb_eval

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

            logger.info(
                "Start training lightgbm model, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            num_boost_round = params.pop("num_boost_round")
            early_stopping_rounds = params.pop("early_stopping_rounds")

            self._model = lgb.train(
                params=params,
                train_set=lgb_train,
                init_model=init_model_path,
                num_boost_round=num_boost_round,
                valid_sets=lgb_eval,
                categorical_feature=self._categorical_list,
                early_stopping_rounds=early_stopping_rounds,
                fobj=obj_function,
                feval=eval_function,
                verbose_eval=False,
            )

            params["num_boost_round"] = num_boost_round
            params["early_stopping_rounds"] = early_stopping_rounds

            logger.info(
                "Training lightgbm model finished, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

        else:
            raise ValueError("Model parameters is None.")
        self.count += 1

    def _binary_increment(self, train_dataset: BaseDataset, **entity):
        """
        This method is used to train lightgbm (booster)
        model in binary classification.
        :param train_dataset: new boosting train dataset.
        :return: None
        """
        decay_rate = self._decay_rate
        init_model_path = os.path.join(self._model_save_root, self._model_file_name)
        assert os.path.isfile(init_model_path)

        assert self._train_flag == ConstantValues.increment
        assert self._task_name == ConstantValues.binary_classification

        lgb_entity = self.__lgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=None,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        dataset_bunch = lgb_entity.train_dataset
        init_model = lgb.Booster(model_file=init_model_path)
        assert 0 < decay_rate < 1, "Value: decay_rate must in (0, 1), but get {} instead.".format(decay_rate)
        self._model = init_model.refit(data=dataset_bunch.data,
                                       label=dataset_bunch.target,
                                       decay_rate=decay_rate)

    def _multiclass_increment(self, train_dataset: BaseDataset, **entity):
        """
        This method is used to train lightgbm (booster)
        model in multiclass classification.
        :param train_dataset: new boosting train dataset.
        :return: None
        """
        """
        This method is used to train lightgbm (booster)
        model in binary classification.
        :param train_dataset: new boosting train dataset.
        :return: None
        """
        decay_rate = self._decay_rate
        init_model_path = os.path.join(self._model_save_root, self._model_file_name)
        assert os.path.isfile(init_model_path)

        assert self._train_flag == ConstantValues.increment
        assert self._task_name == ConstantValues.binary_classification

        lgb_entity = self.__lgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=None,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        dataset_bunch = lgb_entity.train_dataset
        init_model = lgb.Booster(model_file=init_model_path)
        assert 0 < decay_rate < 1, "Value: decay_rate must in (0, 1), but get {} instead.".format(decay_rate)

        self._model = init_model.refit(data=dataset_bunch.data,
                                       label=dataset_bunch.target,
                                       decay_rate=decay_rate)

    def _regression_increment(self, train_dataset: BaseDataset, **entity):
        """
        This method is used to train lightgbm (booster)
        model in regression.
        :param train_dataset: new boosting train dataset.
        :return: None
        """
        """
        This method is used to train lightgbm (booster)
        model in binary classification.
        :param train_dataset: new boosting train dataset.
        :return: None
        """
        decay_rate = self._decay_rate
        init_model_path = os.path.join(self._model_save_root, self._model_file_name)
        assert os.path.isfile(init_model_path)

        assert self._train_flag == ConstantValues.increment
        assert self._task_name == ConstantValues.binary_classification

        lgb_entity = self.__lgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=None,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        dataset_bunch = lgb_entity.train_dataset
        init_model = lgb.Booster(model_file=init_model_path)
        assert 0 < decay_rate < 1, "Value: decay_rate must in (0, 1), but get {} instead.".format(decay_rate)

        self._model = init_model.refit(data=dataset_bunch.data,
                                       label=dataset_bunch.target,
                                       decay_rate=decay_rate)

    def _predict_prob(self, infer_dataset: BaseDataset, **entity):
        assert self._train_flag == ConstantValues.inference
        model_file_path = os.path.join(self._model_save_root, self._model_file_name)
        assert os.path.isfile(model_file_path)

        lgb_entity = self.__lgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                infer_dataset=infer_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        infer_dataset = lgb_entity.infer_dataset
        assert "data" in infer_dataset

        self._model = lgb.Booster(
            model_file=model_file_path
        )

        if self._increment_flag is True:
            inference_result = self._model.predict(data=infer_dataset.data, raw_score=False)
            if self._task_name == ConstantValues.binary_classification:
                inference_result = special.expit(inference_result)
            elif self._task_name == ConstantValues.multiclass_classification:
                inference_result = special.softmax(inference_result)
            else:
                assert self._task_name == ConstantValues.regression
        else:
            inference_result = self._model.predict(data=infer_dataset.data, raw_score=False)
        inference_result = pd.DataFrame({"result": inference_result})
        return inference_result

    def _predict_logit(self, infer_dataset: BaseDataset, **entity):
        assert self._train_flag == ConstantValues.inference

        model_file_path = os.path.join(self._model_save_root, self._model_file_name)
        assert os.path.isfile(model_file_path)

        lgb_entity = self.__lgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                infer_dataset=infer_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        infer_dataset = lgb_entity.infer_dataset
        assert "data" in infer_dataset

        self._model = lgb.Booster(
            model_file=model_file_path
        )

        inference_result = self._model.predict(data=infer_dataset.data, raw_score=True)
        inference_result = pd.DataFrame({"result": inference_result})
        return inference_result

    def _train_preprocess(self):
        pass

    def _predict_preprocess(self):
        pass

    def _eval(self,
              train_dataset: BaseDataset,
              val_dataset: BaseDataset,
              metric: BaseMetric,
              **entity):
        """
        Evaluating
        :param val_dataset: BaseDataset object, used to get validation metric and loss.
        :param train_dataset: BaseDataset object, used to get training metric and loss.
        :param metric: BaseMetric object, used to calculate metric.
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

        lgb_entity = self.__lgb_preprocessing(
            **Bunch(
                label_name=self._target_names,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                check_bunch=self._check_bunch,
                feature_list=self._feature_list,
                categorical_list=self._categorical_list,
                train_flag=self._train_flag,
                task_name=self._task_name
            )
        )

        train_dataset = lgb_entity.train_dataset
        eval_dataset = lgb_entity.eval_dataset

        train_data, train_label, train_target_names = train_dataset.data, train_dataset.target, train_dataset.target_names
        eval_data, eval_label, eval_target_names = eval_dataset.data, eval_dataset.target, eval_dataset.target_names

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
                self._model_file_name
            )
        )

        yaml_write(yaml_dict=self._model_config,
                   yaml_file=os.path.join(
                       self._model_config_root,
                       self._model_config_file_name
                   )
                   )

        yaml_write(yaml_dict={"features": self._feature_list},
                   yaml_file=os.path.join(
                       self._feature_config_root,
                       self._feature_config_file_name
                   )
                   )

    def _update_best(self):
        """
        Do not need to operate.
        :return: None
        """

    def _set_best(self):
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

    def __lgb_preprocessing(self, **params):
        lgb_entity = Bunch()
        if "train_dataset" in params and params[ConstantValues.train_dataset]:
            params["dataset"] = params.pop("train_dataset")
            lgb_train, train_dataset = self.__load_data(**params)
            lgb_entity.lgb_train = lgb_train
            lgb_entity.train_dataset = train_dataset

        if "val_dataset" in params and params[ConstantValues.val_dataset]:
            params["dataset"] = params.pop("val_dataset")
            lgb_eval, eval_dataset = self.__load_data(**params)
            lgb_entity.lgb_eval = lgb_eval
            lgb_entity.eval_dataset = eval_dataset

        if "infer_dataset" in params and params[ConstantValues.infer_dataset]:
            params["dataset"] = params.pop("infer_dataset")
            infer_dataset = self.__load_data(**params)
            lgb_entity.infer_dataset = infer_dataset
        return lgb_entity
