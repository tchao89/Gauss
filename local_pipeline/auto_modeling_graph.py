# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
from __future__ import annotations

import os
from abc import ABC

from typing import List

from local_pipeline.core_chain import CoreRoute
from local_pipeline.preprocess_chain import PreprocessRoute
from local_pipeline.mapping import EnvironmentConfigure
from local_pipeline.base_modeling_graph import BaseModelingGraph

from utils.bunch import Bunch
from utils.check_dataset import check_data
from utils.exception import PipeLineLogicError
from utils.Logger import logger


# This class is used to train model.
class AutoModelingGraph(BaseModelingGraph, ABC):
    def __init__(self, name: str, work_root: str, task_type: str, metric_name: str, train_data_path: str,
                 val_data_path: str = None, feature_configure_path: str = None, target_names: List[str] = None,
                 dataset_type: str = "plain", type_inference: str = "plain", data_clear: str = "plain",
                 feature_generator: str = "featuretools", unsupervised_feature_selector: str = "unsupervised",
                 supervised_feature_selector: str = "supervised", auto_ml: str = "plain", opt_model_names: List[str] = None):

        super().__init__(name, work_root, task_type, metric_name, train_data_path, val_data_path, target_names,
                         feature_configure_path, dataset_type, type_inference, data_clear, feature_generator,
                         unsupervised_feature_selector, supervised_feature_selector, auto_ml, opt_model_names)

        self.already_data_clear = None
        self.best_model = None
        self.best_metric = None
        self.best_result_root = None
        self.best_model_name = None

    def run_route(self,
                  folder_prefix_str,
                  data_clear_flag: bool,
                  feature_generator_flag: bool,
                  unsupervised_feature_generator_flag: bool,
                  supervised_feature_selector_flag: bool,
                  model_zoo: List[str]):

        work_root = self._work_root + "/" + folder_prefix_str

        pipeline_configure = {"data_clear_flag": data_clear_flag,
                              "feature_generator_flag": feature_generator_flag,
                              "unsupervised_feature_selector_flag": unsupervised_feature_generator_flag,
                              "supervised_feature_selector_flag": supervised_feature_selector_flag,
                              "metric_name": self._metric_name,
                              "task_type": self.task_type}

        work_feature_root = work_root + "/feature"
        feature_dict = Bunch()

        feature_dict.user_feature = self._feature_configure_path
        feature_dict.type_inference_feature = os.path.join(work_feature_root, EnvironmentConfigure.feature_dict().type_inference_feature)
        feature_dict.data_clear_feature = os.path.join(work_feature_root, EnvironmentConfigure.feature_dict().data_clear_feature)
        feature_dict.feature_generator_feature = os.path.join(work_feature_root, EnvironmentConfigure.feature_dict().feature_generator_feature)
        feature_dict.unsupervised_feature = os.path.join(work_feature_root, EnvironmentConfigure.feature_dict().unsupervised_feature)
        feature_dict.label_encoding_path = os.path.join(work_feature_root, EnvironmentConfigure.feature_dict().label_encoding_path)
        feature_dict.impute_path = os.path.join(work_feature_root, EnvironmentConfigure.feature_dict().impute_path)
        feature_dict.supervised_feature = os.path.join(work_feature_root, EnvironmentConfigure.feature_dict().supervised_feature)
        feature_dict.final_feature_config = os.path.join(work_feature_root, EnvironmentConfigure.feature_dict().final_feature_config)

        preprocess_chain = PreprocessRoute(name="PreprocessRoute",
                                           feature_path_dict=feature_dict,
                                           task_type=self.task_type,
                                           train_flag=True,
                                           train_data_path=self._train_data_path,
                                           val_data_path=self._val_data_path,
                                           test_data_path=None,
                                           target_names=self._target_names,
                                           dataset_name="plaindataset",
                                           type_inference_name="typeinference",
                                           data_clear_name="plaindataclear",
                                           data_clear_flag=data_clear_flag,
                                           feature_generator_name="featuretools",
                                           feature_generator_flag=feature_generator_flag,
                                           feature_selector_name="unsupervised",
                                           feature_selector_flag=unsupervised_feature_generator_flag)

        try:
            preprocess_chain.run()
        except PipeLineLogicError as e:
            logger.info(e)
            return None

        entity_dict = preprocess_chain.entity_dict
        self.already_data_clear = preprocess_chain.already_data_clear

        assert "dataset" in entity_dict and "val_dataset" in entity_dict

        best_model = None
        best_metric = None
        best_model_name = None
        best_pipeline_config = None

        for model in model_zoo:
            work_model_root = work_root + "/model/" + model + "/"
            model_save_root = work_model_root + "model_save"
            model_config_root = work_model_root + "/model_config"
            feature_config_root = work_model_root + "/feature_config"

            if check_data(already_data_clear=self.already_data_clear, model_name=model) is not True:
                continue

            core_chain = CoreRoute(name="core_route",
                                   train_flag=True,
                                   model_name=model,
                                   model_save_root=model_save_root,
                                   model_config_root=model_config_root,
                                   feature_config_root=feature_config_root,
                                   target_feature_configure_path=feature_dict.final_feature_config,
                                   pre_feature_configure_path=feature_dict.unsupervised_feature,
                                   label_encoding_path=feature_dict.label_encoding_path,
                                   model_type="tree_model",
                                   metrics_name=self._metric_name,
                                   task_type=self.task_type,
                                   feature_selector_name="supervised_selector",
                                   feature_selector_flag=supervised_feature_selector_flag,
                                   auto_ml_type="auto_ml",
                                   opt_model_names=self._opt_model_names,
                                   auto_ml_path="/configure_files/automl_params",
                                   selector_config_path="/configure_files/selector_params")

            core_chain.run(**entity_dict)
            local_metric = core_chain.optimal_metrics
            local_model = core_chain.optimal_model

            if best_model is None:
                best_model = local_model
            if best_metric is None:
                best_metric = local_metric
            if best_model_name is None:
                best_model_name = model
            if best_pipeline_config is None:
                best_pipeline_config = pipeline_configure

            if best_metric is None or best_metric.__cmp__(local_metric) < 0:
                best_model = local_model
                best_metric = local_metric
                best_model_name = model
                best_pipeline_config = pipeline_configure

        return best_model, best_metric, work_root, pipeline_configure

    def _run(self):
        local_result = self.run_route(
            folder_prefix_str="no-clear_feagen_no-unsupfeasel_no-supfeasel",
            data_clear_flag=False,
            feature_generator_flag=True,
            unsupervised_feature_generator_flag=False,
            supervised_feature_selector_flag=False,
            model_zoo=["lightgbm"])

        if local_result is not None:
            self.update_best(*local_result)

        local_result = self.run_route("clear_feagen_supfeasel_no-supfeasel", True, True, True, False, ["lightgbm"])

        if local_result is not None:
            self.update_best(*local_result)

        local_result = self.run_route("no-clear_feagen_supfeasel_supfeasel", False, True, True, True, ["lightgbm"])

        if local_result is not None:
            self.update_best(*local_result)

        local_result = self.run_route("no-clear_no-feagen_no-supfeasel_no-supfeasel", False, False, False, False, ["lightgbm"])

        if local_result is not None:
            self.update_best(*local_result)

        local_result = self.run_route("no-clear_no-feagen_no-supfeasel_supfeasel", False, False, False, True, ["lightgbm"])

        if local_result is not None:
            self.update_best(*local_result)
