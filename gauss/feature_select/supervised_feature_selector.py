# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import os
import json

from entity.base_dataset import BaseDataset
from entity.plain_dataset import PlaintextDataset
from entity.gbdt import GaussLightgbm
from entity.udf_metric import AUC
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
        :param label_encoding_configure_path: label encoding configure path is
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
        self.selector_trial_num = 10
        # default parameters concludes tree selector parameters and gradient parameters.
        # format: {"gradient_feature_selector": {"order": 4, "n_epochs": 100},
        # "GBDTSelector": {"lgb_params": {}, "eval_ratio", 0.3, "importance_type": "gain", "early_stopping_rounds": 100}}
        self._search_space = None
        self._default_parameters = None
        # generated parameters
        self._new_parameters = None
        self._task_name = params["task_name"]
        # # 训练机器学习模型，特征选择模型直接写入该方法中
        # self.hyperopt_ml = TabularAutoML(name="HyperOptAutoMl",
        #                                  train_flag=params["train_flag"],
        #                                  enable=True,
        #                                  opt_model_names=["tpe", "random_search", "anneal", "evolution"],
        #                                  auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config")

    def _train_run(self, **entity):
        """

        :param entity: input dataset, metrics
        :return:
        """

        dataset = entity["dataset"]
        self.set_default_params()
        self.set_search_space()

        selector_tuner = EvolutionTuner()
        for model_name in self.feature_selector_names:
            # 梯度特征选择
            if model_name == "gradient_feature_selector":
                # 接受默认参数列表
                self._new_parameters = self._default_parameters["gradient_feature_selector"]
                # 设定搜索空间
                search_space = self._search_space["gradient_feature_selector"]
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
                    params.update(receive_params)
                    feature_list = self._gradient_based_selector(dataset=dataset, params=params)

                    print("this is a test:", feature_list)
                    # 将data和target包装成为PlainDataset对象
                    data = dataset.feature_choose(feature_list)
                    target = dataset.get_dataset().target
                    data_pair = Bunch(data=data, target=target)
                    train_dataset = PlaintextDataset(name="train_data", task_type="train", data_pair=data_pair)

                    metrics_factory = MetricsFactory()
                    metrics_params = Bunch(name="auc", label_name=dataset.get_dataset().target_names[0])
                    metrics = metrics_factory.get_entity(entity_name=self._metrics_name, **metrics_params)

                    model = GaussLightgbm(name='lightgbm', model_path='./model.txt', train_flag=True, task_type='classification')
                    model_tuner.run(model=model, dataset=train_dataset, metrics=metrics)

                    selector_tuner.receive_trial_result(trial, receive_params, metrics)

    def _predict_run(self, **entity):
        pass

    @property
    def search_space(self):
        return self.search_space()

    def set_search_space(self):
        search_space_path = os.path.join(self._selector_config_path, "search_space.json")
        with open(search_space_path, 'r') as json_file:
            self._search_space = json.load(json_file)

    def _tree_based_selector(self, dataset: BaseDataset):
        assert self._new_parameters is not None

        data = dataset.get_dataset().data
        target = dataset.get_dataset().target

        columns = data.shape[1]

        def len_features(col_ratio: float):
            return int(columns * col_ratio)

        params = self._new_parameters["GBDTSelector"]
        params["topk"]["_value"] = map(len_features, params["topk"]["_value"])

        selector = GBDTSelector()
        selector.fit(data, target,
                     lgb_params=params["lgb_params"],
                     eval_ratio=params["eval_ratio"],
                     early_stopping_rounds=params["early_stopping_rounds"],
                     importance_type=params["importance_type"],
                     num_boost_round=params["num_boost_round"])

        return selector.get_selected_features(topk=params["topk"])

    def _gradient_based_selector(self, dataset: BaseDataset, params: dict):
        # 注意定义n_features
        assert self._new_parameters is not None

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
                                           verbose=1)

        selector.fit(data, target.values.flatten())
        return selector.get_selected_features()

    @property
    def default_params(self):
        return self._default_parameters

    def set_default_params(self):
        default_params_path = os.path.join(self._selector_config_path, "default_parameters.json")
        with open(default_params_path, 'r') as json_file:
            self._default_parameters = json.load(json_file)
