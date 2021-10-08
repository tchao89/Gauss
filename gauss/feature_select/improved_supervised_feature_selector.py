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

from entity.model.single_process_model import SingleProcessModelWrapper
from entity.model.multiprocess_model import MultiprocessModelWrapper
from entity.metrics.base_metric import MetricResult

from gauss.feature_select.base_feature_selector import BaseFeatureSelector

import core.lightgbm as lgb
from core.nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

from utils.base import get_current_memory_gb
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
        self.selector_trial_num = params["selector_trial_num"]
        self.__improved_selector_configure_path = params["improved_selector_configure_path"]
        self.__feature_model_trial = 15
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

        feature_importance_pair = self.__feature_select(**entity)

        original_dataset = entity["train_dataset"]
        original_val_dataset = entity["val_dataset"]
        feature_configure = entity["feature_configure"]
        metric = entity["metric"]
        loss = entity["loss"]

        self._optimize_mode = metric.optimize_mode
        columns = original_dataset.get_dataset().data.shape[1]

        # 创建自动机器学习对象
        model_tuner = entity["auto_ml"]
        model_tuner.is_final_set = False

        model = entity["model"]
        assert isinstance(model, (SingleProcessModelWrapper, MultiprocessModelWrapper))

        selector_tuner = HyperoptTuner(
            algorithm_name="tpe",
            optimize_mode=self._optimize_mode
        )

        default_params_path = os.path.join(self.__improved_selector_configure_path, "default_parameters.json")
        with open(default_params_path, 'r', encoding="utf-8") as json_file:
            parameters = json.load(json_file)

        search_space_path = os.path.join(self.__improved_selector_configure_path, "search_space.json")
        with open(search_space_path, 'r', encoding="utf-8") as json_file:
            search_space = json.load(json_file)

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

            metric.label_name = original_dataset.get_dataset().target_names[0]

            logger.info(
                "Parse feature configure and generate feature configure object, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            feature_configure.file_path = self._feature_configure_path

            feature_configure.parse(method="system")
            feature_configure.feature_select(feature_list=feature_list)

            logger.info(
                "Auto model training starts, with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            # 返回训练好的最佳模型
            model_tuner.run(
                model=model,
                train_dataset=original_dataset,
                val_dataset=original_val_dataset,
                metric=metric,
                loss=loss,
                feature_configure=feature_configure
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

        if model_tuner.is_final_set is False:
            model.set_best_model()

        self._optimal_metric = model.val_best_metric_result.result

        # save features
        self._final_feature_names = model.feature_list
        if isinstance(original_dataset.get_dataset().data, pd.DataFrame):
            self.final_configure_generation()
        else:
            assert isinstance(original_dataset.get_dataset().data, np.ndarray)
            self.multiprocess_final_configure_generation()

    def __feature_select(self, **entity):
        train_dataset = entity["train_dataset"].get_dataset()
        val_dataset = entity["val_dataset"].get_dataset()

        train_dataset = lgb.Dataset(train_dataset.data, train_dataset.target, feature_name=train_dataset.data.columns)
        val_dataset = lgb.Dataset(val_dataset.data, val_dataset.target, feature_name=val_dataset.data.columns)

        tuner = HyperoptTuner(
            algorithm_name="tpe",
            optimize_mode="minimize"
        )

        default_params_path = os.path.join(self._auto_ml_path, "default_parameters.json")
        with open(default_params_path, 'r', encoding="utf-8") as json_file:
            default_parameters = json.load(json_file)

        search_space_path = os.path.join(self._auto_ml_path, "search_space.json")
        with open(search_space_path, 'r', encoding="utf-8") as json_file:
            search_space = json.load(json_file)

        tuner.update_search_space(search_space=search_space.get("lightgbm"))

        selector = None
        for trial in range(self.__feature_model_trial):
            if default_parameters is not None:
                params = default_parameters.get("lightgbm")
                assert params is not None

                receive_params = tuner.generate_parameters(trial)
                params.update(receive_params)
                params["lgb_params"]["objective"] = "binary"
                params["lgb_params"]["metric"] = "binary_logloss"

                eval_result = dict()

                selector = lgb.train(params=params["lgb_params"],
                                     train_set=train_dataset,
                                     valid_sets=val_dataset,
                                     num_boost_round=params["num_boost_round"],
                                     early_stopping_rounds=params["early_stopping_rounds"],
                                     callbacks=lgb.record_evaluation(eval_result=eval_result),
                                     verbose_eval=False)

                metric_result = min(eval_result["eval"]["logloss"])
                tuner.receive_trial_result(trial, receive_params, metric_result)
            else:
                raise ValueError("Default parameters is None.")

        assert isinstance(selector, lgb.Booster)
        feature_name_list = selector.feature_name()
        importance_list = list(selector.feature_importance())
        feature_importance_pair = [(fe, round(im, 2)) for fe, im in zip(feature_name_list, importance_list)]
        feature_importance_pair = sorted(feature_importance_pair, key=lambda x: x[1], reverse=True)

        return feature_importance_pair

    def _increment_run(self, **entity):
        self._train_run(**entity)

    @property
    def optimal_metric(self):
        """
        Get optimal metric.
        :return:
        """
        return self._optimal_metric

    @classmethod
    def update_feature_conf(cls, feature_conf, feature_list):
        """
        Update feature configure dict.
        :param feature_conf:
        :param feature_list:
        :return:
        """
        for feature in feature_conf.keys():
            if feature_conf[feature]["index"] not in feature_list:
                feature_conf[feature]["used"] = False

        return feature_conf

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

    def multiprocess_final_configure_generation(self):
        """
        Write configure file in multiprocess mode.
        :return:
        """
        feature_conf = yaml_read(yaml_file=self._feature_configure_path)
        logger.info("final_feature_names: %s", str(self._final_feature_names))

        for item in feature_conf.keys():
            if feature_conf[item]["index"] not in self._final_feature_names:
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
        """
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
