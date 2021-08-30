# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
from gauss_factory.abstarct_guass import AbstractGauss

from gauss.data_clear.plain_data_clear import PlainDataClear
from gauss.feature_generation.featuretools_generation import FeatureToolsGenerator
from gauss.feature_select.supervised_feature_selector import SupervisedFeatureSelector
from gauss.feature_select.unsupervised_feature_selector import UnsupervisedFeatureSelector
from gauss.type_inference.plain_type_inference import PlainTypeInference
from gauss.auto_ml.tabular_auto_ml import TabularAutoML

class ComponentFactory(AbstractGauss):

    def get_entity(self, entity_name: str):
        return None

    def get_component(self, component_name: str, **params):

        if component_name.lower() == "plaindataclear":
            # name: str, train_flag: bool, enable: bool, model_name: str, feature_configure_path: str, strategy_dict
            return PlainDataClear(**params)

        if component_name.lower() == "featuretoolsgeneration":
            # name: str, train_flag: bool, enable: bool, feature_config_path: str, label_encoding_configure_path: str
            return FeatureToolsGenerator(**params)

        if component_name.lower() == "supervisedfeatureselector":
            # name: str, train_flag: bool, enable: bool, feature_config_path: str, label_encoding_configure_path: str,
            # task_name: str, selector_config_path: str, metrics_name: str
            return SupervisedFeatureSelector(**params)

        if component_name.lower() == "unsupervisedfeatureselector":
            # name: str, train_flag: bool, enable: bool, feature_config_path: str,
            # label_encoding_configure_path: str, feature_select_configure_path: str
            return UnsupervisedFeatureSelector(**params)

        if component_name.lower() == "plaintypeinference":
            # name: str, task_name: str, train_flag: bool, source_file_path="null",
            # final_file_path: str, final_file_prefix="final"
            return PlainTypeInference(**params)

        if component_name.lower() == "tabularautoml":
            # name: str, train_flag: bool, enable: bool, opt_model_names, auto_ml_path
            return TabularAutoML(**params)

        return None
