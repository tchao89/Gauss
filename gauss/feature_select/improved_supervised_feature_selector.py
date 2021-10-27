"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
Training model with supervised feature selector.
"""
import os
import json

import numpy as np
import pandas as pd

from entity.metrics.base_metric import MetricResult
from entity.model.model import ModelWrapper

from gauss.feature_select.base_feature_selector import BaseFeatureSelector

import core.lightgbm as lgb
from core.nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

from utils.base import get_current_memory_gb
from utils.bunch import Bunch
from utils.yaml_exec import yaml_read
from utils.yaml_exec import yaml_write
from utils.Logger import logger
from utils.constant_values import ConstantValues


class ImprovedSupervisedFeatureSelector(BaseFeatureSelector):
    """
    SupervisedFeatureSelector object.
    """

    def __init__(self, **params):
        """
        :param name: Name of this operator.
        :param train_flag: It is a bool value, and if it is True,
        this operator will be used for training, and if it is False, this operator will be
        used for predict.
        :param enable: It is a bool value, and if it is True, this operator will be used.
        :param feature_config_path: Feature config path is a path from yaml file which is
        generated from type inference operator.
        :param label_encoding_configure_path:
        :param task_name: string object
        :param selector_config_path: root path of supervised selector configure files
        :param metrics_name: Construct BaseMetric object by entity factory.
        """
        assert ConstantValues.model_name in params
        assert ConstantValues.auto_ml_path in params
        assert ConstantValues.metric_name in params

        super().__init__(
            name=params[ConstantValues.name],
            train_flag=params[ConstantValues.train_flag],
            enable=params[ConstantValues.enable],
            task_name=params[ConstantValues.task_name],
            feature_configure_path=params[ConstantValues.feature_configure_path]
        )

        self._metrics_name = params[ConstantValues.metric_name]
        self._model_name = params[ConstantValues.model_name]
        self._auto_ml_path = params[ConstantValues.auto_ml_path]
        self._model_root_path = params[ConstantValues.model_root_path]
        self._final_file_path = params[ConstantValues.final_file_path]

        self._optimize_mode = None

        # max trail num for selector tuner
        self.selector_trial_num = params[ConstantValues.selector_trial_num]
        self.__improved_selector_configure_path = params[ConstantValues.improved_selector_configure_path]
        self.__feature_model_trial = params[ConstantValues.feature_model_trial]
        # default parameters concludes tree selector parameters and gradient parameters.
        # format: {"gradient_feature_selector": {"order": 4, "n_epochs": 100},
        # "GBDTSelector": {"lgb_params": {}, "eval_ratio", 0.3, "importance_type":
        # "gain", "early_stopping_rounds": 100}}
        self._search_space = None
        self._default_parameters = None
        self._final_feature_names = None

        self._optimal_metrics = None

        self.__set_default_params()
        self.__set_search_space()

    def __train_selector(self, **entity):
        assert "train_dataset" in entity.keys()
        assert "val_dataset" in entity.keys()

        # use auto ml component to train a lightgbm model and get feature_importance_pair
        selector_model_tuner = entity["selector_auto_ml"]
        feature_configure = entity["feature_configure"]

        feature_configure.file_path = self._feature_configure_path
        feature_configure.parse(method="system")
        entity[ConstantValues.selector_model].update_feature_conf(feature_conf=feature_configure)

        selector_entity = Bunch(model=entity["selector_model"],
                                feature_configure=entity["feature_configure"],
                                train_dataset=entity["train_dataset"],
                                val_dataset=entity["val_dataset"],
                                metric=entity["selector_metric"],
                                loss=entity["loss"])
        selector_model_tuner.run(**selector_entity)

        selector_entity["model"].set_best_model()
        selector = selector_entity["model"].model

        assert isinstance(selector, lgb.Booster)
        feature_name_list = selector.feature_name()
        importance_list = list(selector.feature_importance())
        feature_importance_pair = [(fe, round(im, 2)) for fe, im in zip(feature_name_list, importance_list)]
        feature_importance_pair = sorted(feature_importance_pair, key=lambda x: x[1], reverse=True)
        return feature_importance_pair

    def _train_run(self, **entity):
        """
        feature_select
        :param entity: input dataset, metric
        :return: None
        """
        logger.info(
            "Starting training supervised selectors, with current memory usage: %.2f GiB",
            get_current_memory_gb()["memory_usage"]
        )

        assert "train_dataset" in entity.keys()
        assert "val_dataset" in entity.keys()
        assert "model" in entity.keys()
        assert "metric" in entity.keys()
        assert "auto_ml" in entity.keys()
        assert "feature_configure" in entity.keys()
        assert "loss" in entity.keys()
        assert "selector_model" in entity.keys()
        assert "selector_auto_ml" in entity.keys()
        assert "selector_metric" in entity.keys()

        feature_importance_pair = self.__train_selector(**entity)

        train_dataset = entity[ConstantValues.train_dataset]
        val_dataset = entity[ConstantValues.val_dataset]

        feature_configure = entity[ConstantValues.feature_configure]
        metric = entity[ConstantValues.metric]
        loss = entity[ConstantValues.loss]

        self._optimize_mode = metric.optimize_mode
        columns = train_dataset.get_dataset().data.shape[1]

        # 创建自动机器学习对象
        model_tuner = entity[ConstantValues.auto_ml]
        model_tuner.automl_final_set = False

        model = entity[ConstantValues.model]
        assert isinstance(model, ModelWrapper)

        selector_tuner = HyperoptTuner(
            algorithm_name="tpe",
            optimize_mode=self._optimize_mode
        )

        search_space = self._search_space
        parameters = self._default_parameters

        # 更新特征选择模块的搜索空间
        logger.info(
            "Update search space for supervised selector, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )
        selector_tuner.update_search_space(search_space=search_space)

        logger.info(
            "Starting training supervised selector models, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        for trial in range(self.selector_trial_num):
            logger.info(
                "supervised selector models training, round: {:d}, "
                "with current memory usage: {:.2f} GiB".format(
                    trial, get_current_memory_gb()["memory_usage"]
                )
            )

            receive_params = selector_tuner.generate_parameters(trial)
            # feature selector hyper-parameters
            parameters.update(receive_params)

            def len_features(col_ratio: float):
                return int(columns * col_ratio)

            parameters["topk"] = len_features(parameters["topk"])
            feature_list = [item[0] for item in feature_importance_pair]
            logger.info(
                "trial: {:d}, supervised selector training, and starting training model, "
                "with current memory usage: {:.2f} GiB".format(
                    trial,
                    get_current_memory_gb()["memory_usage"]
                )
            )

            metric.label_name = train_dataset.get_dataset().target_names[0]

            logger.info(
                "Parse feature configure and generate feature configure object, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            feature_configure.file_path = self._feature_configure_path

            feature_configure.parse(method="system")
            feature_configure.feature_select(feature_list=feature_list,
                                             use_index_flag=False)
            entity[ConstantValues.model].update_feature_conf(feature_conf=feature_configure)

            logger.info(
                "Auto model training starts, with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            # 返回训练好的最佳模型
            model_tuner.run(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                metric=metric,
                loss=loss
            )

            assert isinstance(model.val_metric, MetricResult)
            local_optimal_metric = model_tuner.local_best

            logger.info(
                "Receive supervised selectors training trial result, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            selector_tuner.receive_trial_result(
                trial,
                receive_params,
                local_optimal_metric.result
            )

        if model_tuner.automl_final_set is False:
            model.set_best_model()

        self._optimal_metric = model.val_best_metric_result.result

        # save features
        self._final_feature_names = model.feature_list
        if isinstance(train_dataset.get_dataset().data, pd.DataFrame):
            self.final_configure_generation()
        else:
            raise TypeError(
                "Training data must be type: pd.Dataframe but get {} instead".format(
                    type(train_dataset.get_dataset().data))
            )

    def _increment_run(self, **entity):
        pass

    def _predict_run(self, **entity):
        pass

    def final_configure_generation(self):
        """
        Write configure file
        :return:
        """
        feature_conf = yaml_read(yaml_file=self._feature_configure_path)
        logger.info("final_feature_names: %s", str(self._final_feature_names))
        for item in feature_conf.keys():
            if item not in self._final_feature_names:
                feature_conf[item]["used"] = False

        yaml_write(yaml_file=self._final_file_path, yaml_dict=feature_conf)

    @property
    def search_space(self):
        """
        Get search space.
        :return:
        """
        assert self._search_space is not None
        return self._search_space

    def __set_search_space(self):
        """
        Read search space file.
        :return:
        """
        search_space_path = os.path.join(self.__improved_selector_configure_path, "search_space.json")
        with open(search_space_path, 'r', encoding="utf-8") as json_file:
            self._search_space = json.load(json_file)

    @classmethod
    def __load_search_space(cls, json_dict: dict, res=None):
        """
        Read search space configuration.
        :param json_dict: It's a json dict that need to be recursion.
        :param res: result dict that has been nested dismissed.
        :return: dict
        """
        if res is None:
            res = {}
        for key in json_dict.keys():
            key_value = json_dict.get(key)
            if isinstance(key_value, dict) and \
                    "_type" not in key_value.keys() and \
                    "_value" not in key_value.keys():
                cls.__load_search_space(key_value, res)
            else:
                res[key] = key_value
        return res

    @property
    def default_params(self):
        """
        Get default parameters.
        :return:
        """
        return self._default_parameters

    def __set_default_params(self):
        """
        Read default parameters.
        :return: None
        """
        default_params_path = os.path.join(
            self.__improved_selector_configure_path,
            "default_parameters.json"
        )

        with open(default_params_path, 'r', encoding="utf-8") as json_file:
            self._default_parameters = json.load(json_file)

    @classmethod
    def __load_default_params(cls, json_dict: dict, res=None):
        """a
        Read default parameters.
        :param json_dict:
        :param res:
        :return:
        """
        if res is None:
            res = {}
        for key in json_dict.keys():
            key_value = json_dict.get(key)
            if isinstance(key_value, dict):
                cls.__load_default_params(key_value, res)
            else:
                res[key] = key_value
        return res

    @classmethod
    def __check_dataset(cls, dataframe: pd.DataFrame):
        """
        check dataset and remove irregular columns,
        if there is existing at least a features containing
        np.nan, np.inf or -np.inf, this method will return False.
        :param dataframe:
        :return: bool
        """
        indices_to_keep = dataframe.isin([np.nan, np.inf, -np.inf]).any()
        features = indices_to_keep[indices_to_keep is True].index
        if not list(features):
            return True

        return False
