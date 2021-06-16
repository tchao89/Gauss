# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
from gauss.feature_select.base_feature_selector import BaseFeatureSelector
from gauss.auto_ml.tabular_auto_ml import TabularAutoML

from core.nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector
from core.nni.algorithms.feature_engineering.gbdt_selector import GBDTSelector


class SupervisedFeatureSelector(BaseFeatureSelector):

    def __init__(self, name, train_flag, enable, feature_config_path, label_encoding_configure_path):
        super(SupervisedFeatureSelector, self).__init__(name=name,
                                                        train_flag=train_flag,
                                                        enable=enable,
                                                        feature_configure_path=feature_config_path)

        self.feature_selector_names = ["gradient_feature_selector", "GBDTSelector"]
        self.feature_selector_models = []
        self._default_parameters = None
        self.hyperopt_ml = TabularAutoML(name="HyperOptAutoMl", train_flag=train_flag, enable=True, opt_model_names=["tpe", "random_search", "anneal", "evolution"])

    def _train_run(self, **entity):
        # 设定机器学习模型的的初始参数
        self.hyperopt_ml.default_params = self.default_params
        
        hyperopt_ml.run(entity)

    def _predict_run(self, **entity):
        pass

    def _get_search_space(self):
        pass

    def _tree_based_selector(self):
        pass

    def _gradient_based_selector(self):
        pass

    @property
    def default_params(self):
        return self._default_parameters

    @default_params.setter
    def default_params(self, params: dict):
        self._default_parameters = params
