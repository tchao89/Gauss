# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
"""
Training model with supervised feature selector.
"""
import copy
import os
import json
from typing import List

import numpy as np
import pandas as pd

from entity.dataset.base_dataset import BaseDataset
from entity.model.single_process_model import SingleProcessModelWrapper
from entity.model.multiprocess_model import MultiprocessModelWrapper
from entity.metrics.base_metric import MetricResult

from gauss.feature_select.base_feature_selector import BaseFeatureSelector

from core.nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector
from core.nni.algorithms.feature_engineering.gbdt_selector import GBDTSelector
from core.nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

from utils.base import get_current_memory_gb
from utils.common_component import yaml_read, yaml_write
from utils.Logger import logger


class SupervisedFeatureSelector(BaseFeatureSelector):
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
        :param task_name:
        :param selector_config_path:
        :param metrics_name:
        """
        assert "model_name" in params
        assert "auto_ml_path" in params
        assert "metrics_name" in params
        assert "model_save_path" in params

        super().__init__(
            name=params["name"],
            train_flag=params["train_flag"],
            enable=params["enable"],
            task_name=params["task_name"],
            feature_configure_path=params["feature_config_path"]
        )

        self._label_encoding_configure_path = params["label_encoding_configure_path"]
        self._metrics_name = params["metrics_name"]
        self._selector_config_path = params["selector_config_path"]
        self._model_name = params["model_name"]
        self._auto_ml_path = params["auto_ml_path"]
        self._model_save_path = params["model_save_path"]
        self._final_file_path = params["final_file_path"]

        self._model_config_root = params["model_config_root"]
        self._feature_config_root = params["feature_config_root"]

        self._optimize_mode = None

        # selector names
        self._feature_selector_names = params["feature_selector_model_names"]
        # max trail num for selector tuner
        self.selector_trial_num = params["selector_trial_num"]
        # default parameters concludes tree selector parameters and gradient parameters.
        # format: {"gradient_feature_selector": {"order": 4, "n_epochs": 100},
        # "GBDTSelector": {"lgb_params": {}, "eval_ratio", 0.3, "importance_type":
        # "gain", "early_stopping_rounds": 100}}
        self._search_space = None
        self._default_parameters = None
        self._final_feature_names = None
        # generated parameters
        self._new_parameters = None
        self._task_name = params["task_name"]

        self._optimal_model = None
        self._optimal_metrics = None

        if not self._train_flag:
            self._result = None

    @property
    def feature_selector_names(self):
        """
        Get feature selector name.
        :return:
        """
        return self._feature_selector_names

    @feature_selector_names.setter
    def feature_selector_names(self, selector_names: List[str]):
        """
        set feature selector name.
        :param selector_names:
        :return:
        """
        for item in selector_names:
            assert item in ["GBDTSelector", "gradient_feature_selector"]
        self._feature_selector_names = selector_names

    def choose_selector(self, selector_name: str, dataset: BaseDataset, params: dict):
        """
        Get selector model
        :param selector_name:
        :param dataset:
        :param params:
        :return:
        """
        if selector_name == "gradient_feature_selector":
            return self._gradient_based_selector(dataset=dataset, params=params)
        if selector_name == "GBDTSelector":
            return self._tree_based_selector(dataset=dataset, params=params)
        return None

    @classmethod
    def read_search_space(cls, json_dict: dict, res=None):
        """
        Read search space configuration.
        :param json_dict:
        :param res:
        :return:
        """
        if res is None:
            res = {}
        for key in json_dict.keys():
            key_value = json_dict.get(key)
            if isinstance(key_value, dict) and \
                    "_type" not in key_value.keys() and \
                    "_value" not in key_value.keys():
                cls.read_search_space(key_value, res)
            else:
                res[key] = key_value
        return res

    @classmethod
    def read_default_params(cls, json_dict: dict, res=None):
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
                cls.read_default_params(key_value, res)
            else:
                res[key] = key_value
        return res

    def _train_run(self, **entity):
        """
        feature_select
        :param entity: input dataset, metrics
        :return: None
        """
        logger.info(
            "Starting training supervised selectors, with current memory usage: %.2f GiB",
            get_current_memory_gb()["memory_usage"]
        )

        assert "train_dataset" in entity.keys()
        assert "val_dataset" in entity.keys()
        assert "model" in entity.keys()
        assert "metrics" in entity.keys()
        assert "auto_ml" in entity.keys()
        assert "feature_configure" in entity.keys()

        original_dataset = entity["train_dataset"]

        original_val_dataset = entity["val_dataset"]

        logger.info(
            "Loading hyperparameters and search space "
            "for supervised selector, with current memory usage: %.2f GiB",
            get_current_memory_gb()["memory_usage"]
        )
        self.set_default_params()
        self.set_search_space()

        lgb_params = None
        search_space = None

        metrics = entity["metrics"]
        self._optimize_mode = metrics.optimize_mode

        # 创建自动机器学习对象
        model_tuner = entity["auto_ml"]
        model_tuner.is_final_set = False

        model = entity["model"]
        assert isinstance(model, SingleProcessModelWrapper) \
               or isinstance(model, MultiprocessModelWrapper)

        selector_tuner = HyperoptTuner(algorithm_name="tpe", optimize_mode=self._optimize_mode)

        for model_name in self._feature_selector_names:
            # 梯度特征选择
            logger.info(
                "Choose supervised selector: {}, with current memory usage: {:.2f} GiB".format(
                    model_name,
                    get_current_memory_gb()["memory_usage"]
                )
            )

            if model_name == "gradient_feature_selector":
                if self.check_dataset(original_dataset.get_dataset().data):
                    # 接受默认参数列表
                    self._new_parameters = self._default_parameters[model_name]
                    # 设定搜索空间
                    search_space = self._search_space[model_name]
                else:
                    continue

            elif model_name == "GBDTSelector":
                self._new_parameters = self._default_parameters[model_name]

                assert self._new_parameters.get("lgb_params")
                assert isinstance(self._new_parameters, dict)
                lgb_params = self._new_parameters.get("lgb_params")
                lgb_params = lgb_params.keys()

                # flatten dict
                self._new_parameters = self.read_default_params(json_dict=self._new_parameters)

                search_space = self._search_space[model_name]

                search_space = self.read_search_space(json_dict=search_space)

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

                feature_configure = copy.deepcopy(entity["feature_configure"])

                params = self._new_parameters
                receive_params = selector_tuner.generate_parameters(trial)
                # feature selector hyper-parameters
                params.update(receive_params)

                if model_name == "GBDTSelector":
                    lgb_params_dict = {}
                    for key in params.keys():

                        if key in lgb_params:
                            lgb_params_dict[key] = params[key]

                    params["lgb_params"] = lgb_params_dict

                feature_list = self.choose_selector(
                    selector_name=model_name,
                    dataset=original_dataset,
                    params=params
                )
                logger.info(
                    "trial: {:d}, supervised selector training, and starting training model, "
                    "with current memory usage: {:.2f} GiB".format(
                        trial,
                        get_current_memory_gb()["memory_usage"]
                    )
                )

                metrics.label_name = original_dataset.get_dataset().target_names[0]

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
                    "Update hyperparameters of model, "
                    "with current memory usage: {:.2f} GiB".format(
                        get_current_memory_gb()["memory_usage"]
                    )
                )
                model.update_feature_conf(feature_conf=feature_configure)

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
                    metrics=metrics
                )

                assert isinstance(model.val_metrics, MetricResult)
                local_optimal_metrics = model_tuner.local_best

                logger.info(
                    "Receive supervised selectors training trial result, "
                    "with current memory usage: {:.2f} GiB".format(
                        get_current_memory_gb()["memory_usage"]
                    )
                )
                selector_tuner.receive_trial_result(trial, receive_params, local_optimal_metrics.result)

        if model_tuner.is_final_set is False:
            model.set_best_model()

        self._optimal_metrics = model.val_best_metric_result.result

        logger.info("Total trained models: {:d}".format(model.count))
        # save features
        self._final_feature_names = model.feature_list
        if isinstance(original_dataset.get_dataset().data, pd.DataFrame):
            self.final_configure_generation()
        else:
            assert isinstance(original_dataset.get_dataset().data, np.ndarray)
            self.multiprocess_final_configure_generation()

    @property
    def optimal_metrics(self):
        """
        Get optimal metrics.
        :return:
        """
        return self._optimal_metrics

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

    @property
    def result(self):
        """
        Get result.
        :return:
        """
        return self._result

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

    def set_search_space(self):
        """
        Read search space file.
        :return:
        """
        search_space_path = os.path.join(self._selector_config_path, "search_space.json")
        with open(search_space_path, 'r', encoding="utf-8") as json_file:
            self._search_space = json.load(json_file)

    @classmethod
    def _tree_based_selector(cls, dataset: BaseDataset, params: dict):
        data = dataset.get_dataset().data
        target = dataset.get_dataset().target

        columns = data.shape[1]

        def len_features(col_ratio: float):
            return int(columns * col_ratio)

        params["topk"] = len_features(params["topk"])

        selector = GBDTSelector()
        if isinstance(data, pd.DataFrame) and isinstance(target, pd.DataFrame):
            selector.fit(data.values, target.values.flatten(),
                         lgb_params=params["lgb_params"],
                         eval_ratio=params["eval_ratio"],
                         early_stopping_rounds=params["early_stopping_rounds"],
                         importance_type=params["importance_type"],
                         num_boost_round=params["num_boost_round"])
        else:
            selector.fit(data, target.flatten(),
                         lgb_params=params["lgb_params"],
                         eval_ratio=params["eval_ratio"],
                         early_stopping_rounds=params["early_stopping_rounds"],
                         importance_type=params["importance_type"],
                         num_boost_round=params["num_boost_round"])
        return selector.get_selected_features(topk=params["topk"])

    @classmethod
    def _gradient_based_selector(cls, dataset: BaseDataset, params: dict):
        # 注意定义n_features
        data = dataset.get_dataset().data.astype(np.float32)
        target = dataset.get_dataset().target

        columns = data.shape[1]

        def len_features(col_ratio: float):
            return int(columns * col_ratio)

        params["n_features"] = len_features(params["n_features"])

        selector = FeatureGradientSelector(
            order=params["order"],
            n_epochs=params["n_epochs"],
            batch_size=params["batch_size"],
            device=params["device"],
            classification=params["classification"],
            learning_rate=params["learning_rate"],
            n_features=params["n_features"],
            verbose=0
        )
        selector.fit(data, target.values.flatten())
        return selector.get_selected_features()

    @property
    def default_params(self):
        """
        Get default parameters.
        :return:
        """
        return self._default_parameters

    def set_default_params(self):
        """
        Read default parameters.
        :return: None
        """
        default_params_path = os.path.join(
            self._selector_config_path,
            "default_parameters.json"
        )

        with open(default_params_path, 'r', encoding="utf-8") as json_file:
            self._default_parameters = json.load(json_file)

    @classmethod
    def check_dataset(cls, dataframe: pd.DataFrame):
        """
        check dataset and remove irregular columns.
        :param dataframe:
        :return: bool
        """
        indices_to_keep = dataframe.isin([np.nan, np.inf, -np.inf]).any()
        features = indices_to_keep[indices_to_keep is True].index
        if not list(features):
            return True

        return False
