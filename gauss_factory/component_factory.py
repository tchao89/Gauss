"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab
component factory
"""
from gauss_factory.abstarct_guass import AbstractGauss

from gauss.data_clear.plain_data_clear import PlainDataClear
from gauss.feature_generation.featuretools_generation import FeatureToolsGenerator
from gauss.feature_select.supervised_feature_selector import SupervisedFeatureSelector
from gauss.feature_select.unsupervised_feature_selector import UnsupervisedFeatureSelector
from gauss.type_inference.plain_type_inference import PlainTypeInference
from gauss.label_encode.plain_label_encode import PlainLabelEncode
from gauss.auto_ml.tabular_auto_ml import TabularAutoML

class ComponentFactory(AbstractGauss):
    """
    ComponentFactory object
    """
    def get_entity(self, entity_name: str):
        """
        It does not work here
        :param entity_name:
        :return:
        """
        return None

    def get_component(self, component_name: str, **params):
        """
        Get component.
        :param component_name:
        :param params:
        :return:
        """
        if component_name.lower() == "plain_data_clear":
            # name: str,
            # train_flag: bool,
            # enable: bool,
            # task_name" str,
            # model_name: str,
            # feature_configure_path: str,
            # strategy_dict
            return PlainDataClear(**params)

        if component_name.lower() == "featuretools_generation":
            # name: str,
            # train_flag: bool,
            # enable: bool,
            # task_name" str,
            # feature_config_path: str,
            # final_file_path: str
            return FeatureToolsGenerator(**params)

        if component_name.lower() == "plain_label_encoder":
            # name: str,
            # train_flag: bool,
            # enable: bool,
            # task_name" str,
            # feature_config_path: str,
            # final_file_path: str
            return PlainLabelEncode(**params)

        if component_name.lower() == "supervised_feature_selector":
            # name: str,
            # train_flag: bool,
            # enable: bool,
            # feature_config_path: str,
            # label_encoding_configure_path: str,
            # task_name: str,
            # selector_config_path: str,
            # metrics_name: str
            return SupervisedFeatureSelector(**params)

        if component_name.lower() == "unsupervised_feature_selector":
            # name: str,
            # train_flag: bool,
            # enable: bool,
            # task_name" str,
            # feature_config_path: str,
            # label_encoding_configure_path: str,
            # feature_select_configure_path: str
            return UnsupervisedFeatureSelector(**params)

        if component_name.lower() == "plain_type_inference":
            # name: str,
            # task_name: str,
            # train_flag: bool,
            # enable: bool
            # source_file_path="null",
            # final_file_path: str,
            # final_file_prefix="final"
            return PlainTypeInference(**params)

        if component_name.lower() == "tabular_auto_ml":
            # name: str,
            # train_flag: bool,
            # enable: bool,
            # task_name" str,
            # opt_model_names,
            # auto_ml_path
            return TabularAutoML(**params)

        return None
