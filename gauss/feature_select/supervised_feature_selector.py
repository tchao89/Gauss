"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
Training model with supervised feature selector.
"""
import os
import json
from typing import List

import numpy as np
import pandas as pd

from entity.dataset.base_dataset import BaseDataset
from entity.metrics.base_metric import MetricResult
from entity.model.model import ModelWrapper

from gauss.feature_select.base_feature_selector import BaseFeatureSelector

from core.nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector
from core.nni.algorithms.feature_engineering.gbdt_selector import GBDTSelector
from core.nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

from utils.base import get_current_memory_gb
from utils.yaml_exec import yaml_read
from utils.yaml_exec import yaml_write
from utils.Logger import logger
from utils.constant_values import ConstantValues


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
        self._selector_config_path = params[ConstantValues.selector_configure_path]
        self._model_name = params[ConstantValues.model_name]
        self._auto_ml_path = params[ConstantValues.auto_ml_path]
        self._model_root_path = params[ConstantValues.model_root_path]
        self._final_file_path = params[ConstantValues.final_file_path]

        self._optimize_mode = None

        # selector names
        self._feature_selector_names = params[ConstantValues.feature_selector_model_names]
        # max trail num for selector tuner
        self.selector_trial_num = params[ConstantValues.selector_trial_num]
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

    def __choose_features(self, selector_name: str, dataset: BaseDataset, params: dict):
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

    def __load_specified_configure(self, dataset, selector_name):
        assert selector_name in ["GBDTSelector", "gradient_feature_selector"]

        lgb_params = None
        if selector_name == "gradient_feature_selector":
            if self.__check_dataset(dataset.get_dataset().data):
                # 接受默认参数列表
                parameters = self._default_parameters[selector_name]
                # 设定搜索空间
                search_space = self._search_space[selector_name]
            else:
                raise ValueError("There are irregular values "
                                 "such as np.nan, np.inf or -np.inf in dataset, "
                                 "and gradient feature selector can not start.")

        else:
            parameters = self._default_parameters[selector_name]

            assert parameters.get("lgb_params")
            assert isinstance(parameters, dict)

            lgb_params = parameters.get("lgb_params")
            lgb_params = lgb_params.keys()

            # flatten dict
            parameters = self.__load_default_params(json_dict=parameters)
            search_space = self._search_space[selector_name]
            search_space = self.__load_search_space(json_dict=search_space)

        return parameters, search_space, lgb_params

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

        original_dataset = entity["train_dataset"]
        original_val_dataset = entity["val_dataset"]

        logger.info(
            "Loading hyperparameters and search space "
            "for supervised selector, with current memory usage: %.2f GiB",
            get_current_memory_gb()["memory_usage"]
        )

        feature_configure = entity["feature_configure"]

        metric = entity["metric"]
        loss = entity["loss"]
        self._optimize_mode = metric.optimize_mode

        # 创建自动机器学习对象
        model_tuner = entity["auto_ml"]
        model_tuner.is_final_set = False

        model = entity["model"]
        assert isinstance(model, ModelWrapper), \
            "Object: model should be type ModelWrapper, but get {} instead.".format(
                type(model))

        selector_tuner = HyperoptTuner(
            algorithm_name="tpe",
            optimize_mode=self._optimize_mode
        )

        for model_name in self._feature_selector_names:
            # 梯度特征选择
            logger.info(
                "Choose supervised selector: {}, "
                "with current memory usage: {:.2f} GiB".format(
                    model_name,
                    get_current_memory_gb()["memory_usage"]
                )
            )
            parameters, search_space, lgb_params = self.__load_specified_configure(
                selector_name=model_name,
                dataset=original_dataset
            )

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

                if model_name == "GBDTSelector":
                    lgb_params_dict = {}
                    for key in parameters.keys():
                        if key in lgb_params:
                            lgb_params_dict[key] = parameters[key]

                    parameters["lgb_params"] = lgb_params_dict

                feature_list = self.__choose_features(
                    selector_name=model_name,
                    dataset=original_dataset,
                    params=parameters
                )
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
                feature_configure.feature_select(feature_list=feature_list,
                                                 use_index_flag=True)
                entity[ConstantValues.model].update_feature_conf(feature_conf=feature_configure)

                logger.info(
                    "Auto model training starts, with current memory usage: {:.2f} GiB".format(
                        get_current_memory_gb()["memory_usage"]
                    )
                )
                model_tuner.run(
                    model=model,
                    train_dataset=original_dataset,
                    val_dataset=original_val_dataset,
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

        if model_tuner.is_final_set is False:
            model.set_best_model()

        self._optimal_metric = model.val_best_metric_result.result

        # save features
        self._final_feature_names = model.feature_list
        self.final_configure_generation()

    def _increment_run(self, **entity):
        raise RuntimeError("Class: SupervisedFeatureSelector has no increment function.")

    def _predict_run(self, **entity):
        raise RuntimeError("Class: SupervisedFeatureSelector has no predict function.")

    @property
    def optimal_metric(self):
        """
        Get optimal metric.
        :return:
        """
        return self._optimal_metric

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
        search_space_path = os.path.join(self._selector_config_path, "search_space.json")
        with open(search_space_path, 'r', encoding="utf-8") as json_file:
            self._search_space = json.load(json_file)

    @classmethod
    def __load_search_space(cls, json_dict: dict, res=None):
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
                cls.__load_search_space(key_value, res)
            else:
                res[key] = key_value
        return res

    @classmethod
    def _tree_based_selector(cls, dataset: BaseDataset, params: dict):
        data = dataset.get_dataset().data
        target = dataset.get_dataset().target

        columns = data.shape[1]

        def len_features(col_ratio: float):
            return int(columns * col_ratio)

        params["topk"] = len_features(params["topk"])

        selector = GBDTSelector()
        selector.fit(data.values, target.values.flatten(),
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

    def __set_default_params(self):
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
