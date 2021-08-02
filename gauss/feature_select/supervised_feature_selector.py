# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import os
import json
import operator
from typing import List

from entity.dataset.base_dataset import BaseDataset
from entity.model.model import ModelWrapper
from entity.metrics.base_metric import MetricResult
from gauss.feature_select.base_feature_selector import BaseFeatureSelector

from core.nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector
from core.nni.algorithms.feature_engineering.gbdt_selector import GBDTSelector
from core.nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

from utils.common_component import yaml_read, yaml_write


class SupervisedFeatureSelector(BaseFeatureSelector):

    def __init__(self, **params):
        """
        :param name: Name of this operator.
        :param train_flag: It is a bool value, and if it is True,
        this operator will be used for training, and if it is False, this operator will be used for predict.
        :param enable: It is a bool value, and if it is True, this operator will be used.
        :param feature_config_path: Feature config path is a path from yaml file which is generated from type inference operator.
        :param label_encoding_configure_path:
        :param task_name:
        :param selector_config_path:
        :param metrics_name:
        """
        assert "model_name" in params
        assert "auto_ml_path" in params
        assert "metrics_name" in params
        assert "model_save_path" in params

        super(SupervisedFeatureSelector, self).__init__(name=params["name"],
                                                        train_flag=params["train_flag"],
                                                        enable=params["enable"],
                                                        feature_configure_path=params["feature_config_path"])

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
        self._feature_selector_names = ["gradient_feature_selector", "GBDTSelector"]
        # max trail num for selector tuner
        self.selector_trial_num = 4
        # default parameters concludes tree selector parameters and gradient parameters.
        # format: {"gradient_feature_selector": {"order": 4, "n_epochs": 100},
        # "GBDTSelector": {"lgb_params": {}, "eval_ratio", 0.3, "importance_type": "gain", "early_stopping_rounds": 100}}
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
        return self._feature_selector_names

    @feature_selector_names.setter
    def feature_selector_names(self, selector_names: List[str]):
        for item in selector_names:
            assert item in ["gradient_feature_selector", "GBDTSelector"]
        self._feature_selector_names = selector_names

    def choose_selector(self, selector_name: str, dataset: BaseDataset, params: dict):
        if selector_name == "gradient_feature_selector":

            return self._gradient_based_selector(dataset=dataset, params=params)
        elif selector_name == "GBDTSelector":

            return self._tree_based_selector(dataset=dataset, params=params)
        return None

    @classmethod
    def read_search_space(cls, json_dict: dict, res=None):
        if res is None:
            res = {}
        for key in json_dict.keys():
            key_value = json_dict.get(key)
            if isinstance(key_value, dict) and "_type" not in key_value.keys() and "_value" not in key_value.keys():
                cls.read_search_space(key_value, res)
            else:
                res[key] = key_value
        return res

    @classmethod
    def read_default_params(cls, json_dict: dict, res=None):
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
        assert "dataset" in entity.keys()
        assert "val_dataset" in entity.keys()
        assert "model" in entity.keys()
        assert "metrics" in entity.keys()
        assert "auto_ml" in entity.keys()
        assert "feature_configure" in entity.keys()

        original_dataset = entity["dataset"]

        original_val_dataset = entity["val_dataset"]

        feature_configure = entity["feature_configure"]

        self.set_default_params()
        self.set_search_space()

        lgb_params = None
        search_space = None

        metrics = entity["metrics"]
        self._optimize_mode = metrics.optimize_mode

        model = entity["model"]
        assert isinstance(model, ModelWrapper)

        # 创建自动机器学习对象
        model_tuner = entity["auto_ml"]
        model_tuner.is_final_set = False

        selector_tuner = HyperoptTuner(algorithm_name="random_search", optimize_mode=self._optimize_mode)

        for model_name in self._feature_selector_names:
            # 梯度特征选择
            if model_name == "gradient_feature_selector":
                # 接受默认参数列表
                self._new_parameters = self._default_parameters[model_name]
                # 设定搜索空间
                search_space = self._search_space[model_name]

            elif model_name == "GBDTSelector":
                self._new_parameters = self._default_parameters[model_name]

                assert self._new_parameters.get("lgb_params")
                assert isinstance(self._new_parameters.get("lgb_params"), dict)
                lgb_params = self._new_parameters.get("lgb_params").keys()

                # flatten dict
                self._new_parameters = self.read_default_params(json_dict=self._new_parameters)

                search_space = self._search_space[model_name]

                search_space = self.read_search_space(json_dict=search_space)

            # 更新特征选择模块的搜索空间
            selector_tuner.update_search_space(search_space=search_space)

            for trial in range(self.selector_trial_num):

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

                feature_list = self.choose_selector(selector_name=model_name, dataset=original_dataset, params=params)

                metrics.label_name = original_dataset.get_dataset().target_names[0]

                feature_configure.file_path = self._feature_configure_path

                feature_configure.parse(method="system")
                feature_configure.feature_selector(feature_list=feature_list)

                model.update_feature_conf(feature_conf=feature_configure)

                # 返回训练好的最佳模型
                model_tuner.run(model=model, dataset=original_dataset, val_dataset=original_val_dataset, metrics=metrics)

                assert isinstance(model.val_metrics, MetricResult)
                new_metrics = model.val_metrics.result

                selector_tuner.receive_trial_result(trial, receive_params, new_metrics)

        # save features
        self._final_feature_names = model.feature_list
        self.final_configure_generation()

        if model_tuner.is_final_set is False:
            model.final_set()

    @classmethod
    def update_feature_conf(cls, feature_conf, feature_list):
        for feature in feature_conf.keys():
            if feature_conf[feature]["index"] not in feature_list:
                feature_conf[feature]["used"] = False

        return feature_conf

    def _predict_run(self, **entity):
        assert self.train_flag is False

        dataset = entity["dataset"]
        model = entity["model"]

        assert self._model_save_path
        assert self._final_file_path

        self._result = model.predict(dataset)

    @property
    def result(self):
        return self._result

    def final_configure_generation(self):

        feature_conf = yaml_read(yaml_file=self._feature_configure_path)
        for item in feature_conf.keys():
            if item not in self._final_feature_names:
                feature_conf[item]["used"] = False

        yaml_write(yaml_file=self._final_file_path, yaml_dict=feature_conf)

    @property
    def search_space(self):
        assert self._search_space is not None
        return self._search_space

    def set_search_space(self):
        search_space_path = os.path.join(self._selector_config_path, "search_space.json")
        with open(search_space_path, 'r') as json_file:
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
        data = dataset.get_dataset().data
        target = dataset.get_dataset().target

        columns = data.shape[1]

        def len_features(col_ratio: float):
            return int(columns * col_ratio)

        params["n_features"] = len_features(params["n_features"])

        selector = FeatureGradientSelector(order=params["order"],
                                           n_epochs=params["n_epochs"],
                                           batch_size=params["batch_size"],
                                           device=params["device"],
                                           classification=params["classification"],
                                           learning_rate=params["learning_rate"],
                                           n_features=params["n_features"],
                                           verbose=0)

        selector.fit(data, target.values.flatten())
        return selector.get_selected_features()

    @property
    def default_params(self):
        return self._default_parameters

    def set_default_params(self):
        default_params_path = os.path.join(self._selector_config_path, "default_parameters.json")
        with open(default_params_path, 'r') as json_file:
            self._default_parameters = json.load(json_file)
