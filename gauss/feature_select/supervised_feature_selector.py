# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import os
import json

from entity.base_dataset import BaseDataset
from entity.plain_dataset import PlaintextDataset
from entity.gbdt import GaussLightgbm
from gauss.feature_select.base_feature_selector import BaseFeatureSelector
from gauss.auto_ml.tabular_auto_ml import TabularAutoML

from core.nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector
from core.nni.algorithms.feature_engineering.gbdt_selector import GBDTSelector
from core.nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from gauss_factory.entity_factory import MetricsFactory
from utils.bunch import Bunch


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
        super(SupervisedFeatureSelector, self).__init__(name=params["name"],
                                                        train_flag=params["train_flag"],
                                                        enable=params["enable"],
                                                        feature_configure_path=params["feature_config_path"])

        self._label_encoding_configure_path = params["label_encoding_configure_path"]
        self._metrics_name = params["metrics_name"]
        self._selector_config_path = params["selector_config_path"]

        # selector names
        self.feature_selector_names = ["gradient_feature_selector", "GBDTSelector"]
        # max trail num for selector tuner
        self.selector_trial_num = 2
        # default parameters concludes tree selector parameters and gradient parameters.
        # format: {"gradient_feature_selector": {"order": 4, "n_epochs": 100},
        # "GBDTSelector": {"lgb_params": {}, "eval_ratio", 0.3, "importance_type": "gain", "early_stopping_rounds": 100}}
        self._search_space = None
        self._default_parameters = None
        # generated parameters
        self._new_parameters = None
        self._task_name = params["task_name"]

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

        :param entity: input dataset, metrics
        :return:
        """

        dataset = entity["dataset"]

        self.set_default_params()
        self.set_search_space()

        lgb_params = None
        search_space = None

        selector_tuner = EvolutionTuner()
        for model_name in self.feature_selector_names:
            # 梯度特征选择
            if model_name == "gradient_feature_selector":
                # 接受默认参数列表
                self._new_parameters = self._default_parameters[model_name]
                # 设定搜索空间
                search_space = self._search_space[model_name]
            elif model_name == "GBDTSelector":
                self._new_parameters = self._default_parameters[model_name]

                assert self._new_parameters.get("lgb_params")
                assert isinstance(self._new_parameters["lgb_params"], dict)

                lgb_params = self._new_parameters["lgb_params"].keys()
                self._new_parameters = self.read_default_params(json_dict=self._new_parameters)

                search_space = self._search_space[model_name]
                search_space = self.read_search_space(json_dict=search_space)

            # 更新特征选择模块的搜索空间
            selector_tuner.update_search_space(search_space=search_space)
            # 创建自动机器学习对象
            model_tuner = TabularAutoML(name="auto_ml",
                                        train_flag=True,
                                        enable=True,
                                        opt_model_names=["tpe", "random_search", "anneal", "evolution"],
                                        auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config")

            for trial in range(self.selector_trial_num):
                params = self._new_parameters
                receive_params = selector_tuner.generate_parameters(trial)
                # feature selector hyper-parameters
                params.update(receive_params)

                if model_name == "GBDTSelector":
                    lgb_params_dict = {}
                    for key in params.keys():
                        print(key)
                        print("1")

                        if key in lgb_params:
                            lgb_params_dict[key] = params[key]

                    params["lgb_params"] = lgb_params_dict

                feature_list = self.choose_selector(selector_name=model_name, dataset=dataset, params=params)
                # 将data和target包装成为PlainDataset对象
                data = dataset.feature_choose(feature_list)
                target = dataset.get_dataset().target

                data_pair = Bunch(data=data, target=target, target_names=dataset.get_dataset().target_names)
                train_dataset = PlaintextDataset(name="train_data", task_type="train", data_pair=data_pair)

                val_dataset = PlaintextDataset(name="train_data", task_type="train", data_pair=train_dataset.split())

                metrics_factory = MetricsFactory()
                metrics_params = Bunch(name="auc", label_name=dataset.get_dataset().target_names[0])
                metrics = metrics_factory.get_entity(entity_name=self._metrics_name, **metrics_params)

                model = GaussLightgbm(name='lightgbm', model_path='./model.txt', train_flag=True,
                                      task_type='classification')
                model_tuner.run(model=model, dataset=train_dataset, val_dataset=val_dataset, metrics=metrics)
                print("metrics.metrics_result.result: ", metrics.metrics_result.result)
                selector_tuner.receive_trial_result(trial, receive_params, metrics.metrics_result.result)

    def _predict_run(self, **entity):
        pass

    @property
    def search_space(self):
        return self.search_space()

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
        lgb_params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "num_class": 1,
            "metric": "auc",
            "num_leaves": 32,
            "learning_rate": 0.01,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0,
            "max_depth": 9,
            "nthread": -1,
            "lambda_l2": 0.8
        }
        eval_ratio = 0.1
        early_stopping_rounds = 4
        importance_type = 'gain'
        num_boost_round = 1000
        topk = 10

        selector = GBDTSelector()
        print("11111111111111")
        print(params)
        print(params["lgb_params"])
        print(lgb_params)
        selector.fit(data.values, target.values.flatten(),
                     lgb_params=params["lgb_params"],
                     eval_ratio=eval_ratio,
                     early_stopping_rounds=early_stopping_rounds,
                     importance_type=importance_type,
                     num_boost_round=num_boost_round)

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
