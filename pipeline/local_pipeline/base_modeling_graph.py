"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
Abstract object for pipelines.
"""
from __future__ import annotations

import abc

from gauss_factory.gauss_factory_producer import GaussFactoryProducer

from utils.bunch import Bunch
from utils.constant_values import ConstantValues


class BaseModelingGraph:
    """
    BaseModelingGraph object.
    """

    def __init__(self, **params):
        """

        :param name:
        :param work_root:
        :param task_name:
        :param metric_name:
        :param train_data_path:
        :param val_data_path:
        :param target_names:
        :param feature_configure_path:
        :param dataset_name:
        :param type_inference_name:
        :param data_clear_name:
        :param feature_generator_name:
        :param unsupervised_feature_selector_name:
        :param supervised_feature_selector_name:
        :param auto_ml_name:
        :param opt_model_names:
        :param auto_ml_path:
        :param selector_configure_path:
        """
        assert params["opt_model_names"] is not None

        self._attributes_names = Bunch(
            name=params[ConstantValues.name],
            task_name=params[ConstantValues.task_name],
            target_names=params[ConstantValues.target_names],
            metric_eval_used_flag=params[ConstantValues.metric_eval_used_flag]
        )

        self._work_paths = Bunch(
            work_root=params[ConstantValues.work_root],
            train_data_path=params[ConstantValues.train_data_path],
            val_data_path=params[ConstantValues.val_data_path],
            feature_configure_path=params[ConstantValues.feature_configure_path],
            auto_ml_path=params[ConstantValues.auto_ml_path],
            selector_configure_path=params[ConstantValues.selector_configure_path],
            improved_selector_configure_path=params[ConstantValues.improved_selector_configure_path],
            init_model_root=params[ConstantValues.init_model_root]
        )

        self._entity_names = Bunch(
            dataset_name=params[ConstantValues.dataset_name],
            metric_name=params[ConstantValues.metric_name],
            loss_name=params[ConstantValues.loss_name],
            feature_configure_name=params[ConstantValues.feature_configure_name]
        )

        self._component_names = Bunch(
            type_inference_name=params[ConstantValues.type_inference_name],
            data_clear_name=params[ConstantValues.data_clear_name],
            label_encoder_name=params[ConstantValues.label_encoder_name],
            feature_generator_name=params[ConstantValues.feature_generator_name],
            unsupervised_feature_selector_name=params[ConstantValues.unsupervised_feature_selector_name],
            supervised_feature_selector_name=params[ConstantValues.supervised_feature_selector_name],
            improved_supervised_feature_selector_name=params[ConstantValues.improved_supervised_feature_selector_name],
            auto_ml_name=params[ConstantValues.auto_ml_name]
        )

        self._global_values = Bunch(
            dataset_weight_dict=params[ConstantValues.dataset_weight_dict],
            use_weight_flag=params[ConstantValues.use_weight_flag],
            weight_column_name=params[ConstantValues.weight_column_name],
            train_column_name_flag=params[ConstantValues.train_column_name_flag],
            val_column_name_flag=params[ConstantValues.val_column_name_flag],
            data_file_type=params[ConstantValues.data_file_type],
            selector_trial_num=params[ConstantValues.selector_trial_num],
            auto_ml_trial_num=params[ConstantValues.auto_ml_trial_num],
            opt_model_names=params[ConstantValues.opt_model_names],
            supervised_selector_mode=params[ConstantValues.supervised_selector_mode],
            feature_model_trial=params[ConstantValues.feature_model_trial],
            supervised_selector_model_names=params[ConstantValues.supervised_selector_model_names]
        )

        self._flag_dict = Bunch(
            data_clear_flag=params[ConstantValues.data_clear_flag],
            label_encoder_flag=params[ConstantValues.label_encoder_flag],
            feature_generator_flag=params[ConstantValues.feature_generator_flag],
            unsupervised_feature_selector_flag=params[ConstantValues.unsupervised_feature_selector_flag],
            supervised_feature_selector_flag=params[ConstantValues.supervised_feature_selector_flag]
        )

        self._already_data_clear = None
        self._model_need_clear_flag = params[ConstantValues.model_need_clear_flag]
        self._pipeline_configure = None

    @abc.abstractmethod
    def _run_route(self, **params):
        pass

    @classmethod
    def _create_component(cls, component_name: str, **params):
        gauss_factory = GaussFactoryProducer()
        component_factory = gauss_factory.get_factory(choice="component")
        return component_factory.get_component(component_name=component_name, **params)

    @classmethod
    def _create_entity(cls, entity_name: str, **params):
        gauss_factory = GaussFactoryProducer()
        entity_factory = gauss_factory.get_factory(choice="entity")
        return entity_factory.get_entity(entity_name=entity_name, **params)

    def run(self):
        """
        Start training model with pipeline.
        :return:
        """
        self._run()
        self._set_pipeline_config()

    @abc.abstractmethod
    def _run(self):
        pass

    @abc.abstractmethod
    def _set_pipeline_config(self):
        pass

    @abc.abstractmethod
    def _find_best_result(self, train_results):
        pass

    @property
    def pipeline_configure(self):
        """
        This method is used to get pipeline configure.
        :return: dict
        """
        return self._pipeline_configure

    def __generate_work_path(self):
        pass
