# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

import os

from typing import List

from pipeline.core_chain import CoreRoute
from pipeline.preprocess_chain import PreprocessRoute
from pipeline.mapping import EnvironmentConfigure
from pipeline.base import compare, check_data

from utils.common_component import yaml_write
from utils.bunch import Bunch
from utils.exception import PipeLineLogicError


# This class is used to train model.
class AutoModelingTree(object):

    def __init__(self,
                 name: str,
                 work_root: str,
                 task_type: str,
                 metric_name: str,
                 train_data_path: str,
                 val_data_path: str = None,
                 feature_configure_path: str = None,
                 target_names: List[str] = None,
                 dataset_type: str = "plain",
                 type_inference: str = "plain",
                 data_clear: str = "plain",
                 feature_generator: str = "featuretools",
                 unsupervised_feature_selector: str = "unsupervised",
                 supervised_feature_selector: str = "supervised",
                 auto_ml: str = "plain"
                 ):

        self.name = name
        # experiment root path
        self.work_root = work_root
        self.task_type = task_type
        self.metric_name = metric_name
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.target_names = target_names
        self.feature_configure_path = feature_configure_path
        self.dataset_type = dataset_type
        self.type_inference = type_inference
        self.data_clear = data_clear
        self.feature_generator = feature_generator
        self.unsupervised_feature_selector = unsupervised_feature_selector
        self.supervised_feature_selector = supervised_feature_selector
        self.auto_ml = auto_ml
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

        work_root = self.work_root + "/" + folder_prefix_str
        pipeline_configure_path = work_root + "/" + "pipeline/configure.yaml"
        pipeline_configure = {"data_clear_flag": data_clear_flag,
                              "feature_generator_flag": feature_generator_flag,
                              "unsupervised_feature_selector_flag": unsupervised_feature_generator_flag,
                              "supervised_feature_selector_flag": supervised_feature_selector_flag,
                              "metric_name": self.metric_name,
                              "task_type": self.task_type}

        try:
            yaml_write(yaml_file=pipeline_configure_path, yaml_dict=pipeline_configure)
        except FileNotFoundError:
            yaml_write(yaml_file=pipeline_configure_path, yaml_dict=pipeline_configure)

        work_feature_root = work_root + "/feature"
        feature_dict = Bunch()

        feature_dict.user_feature = self.feature_configure_path
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
                                           train_data_path=self.train_data_path,
                                           val_data_path=self.val_data_path,
                                           test_data_path=None,
                                           target_names=self.target_names,
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
            print(e)
            return None

        entity_dict = preprocess_chain.entity_dict
        self.already_data_clear = preprocess_chain.already_data_clear

        assert "dataset" in entity_dict and "val_dataset" in entity_dict

        best_model = None
        best_metric = None
        best_model_name = None

        for model in model_zoo:
            work_model_root = work_root + "/model/" + model + "/"
            model_save_root = work_model_root + "model_save"
            model_config_root = work_model_root + "/model_config"

            if check_data(already_data_clear=self.already_data_clear, model_name=model) is not True:
                continue

            core_chain = CoreRoute(name="core_route",
                                   train_flag=True,
                                   model_name=model,
                                   model_save_root=model_save_root,
                                   model_config_root=model_config_root,
                                   target_feature_configure_path=feature_dict.final_feature_config,
                                   pre_feature_configure_path=feature_dict.unsupervised_feature,
                                   label_encoding_path=feature_dict.label_encoding_path,
                                   model_type="tree_model",
                                   metrics_name=self.metric_name,
                                   task_type=self.task_type,
                                   feature_selector_name="supervised_selector",
                                   feature_selector_flag=supervised_feature_selector_flag,
                                   auto_ml_type="auto_ml",
                                   auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config",
                                   selector_config_path="/home/liangqian/PycharmProjects/Gauss/configure_files/selector_config")

            core_chain.run(**entity_dict)
            local_model = core_chain.optimal_model
            local_metric = core_chain.optimal_metrics

            if best_model is None:
                best_model = local_model
            if best_metric is None:
                best_metric = local_metric
            if best_model_name is None:
                best_model_name = model

            if best_metric is None or (compare(local_metric, best_metric)) < 0:
                best_model = local_model
                best_metric = local_metric
                best_model_name = model

        return best_model, best_metric, work_root, best_model_name

    # local_best_model, local_best_metric, local_best_work_root, local_best_model_name
    def update_best(self, *params):
        if self.best_metric is None or compare(params[1], self.best_metric) < 0:
            self.best_model = params[0]
            self.best_metric = params[1]
            self.best_result_root = params[2]
            self.best_model_name = params[3]

    def run(self):
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

        yaml_dict = {"best_root": self.best_result_root,
                     "best_model_name": self.best_model_name,
                     "work_root": self.work_root,
                     "task_type": self.task_type,
                     "metric_name": self.metric_name,
                     "dataset_name": self.dataset_type,
                     "type_inference": self.type_inference,
                     "data_clear": self.data_clear,
                     "feature_generator": self.feature_generator,
                     "unsupervised_feature_selector": self.unsupervised_feature_selector,
                     "supervised_feature_selector": self.supervised_feature_selector,
                     "auto_ml": self.auto_ml,
                     "best_metric": float(self.best_metric)}

        yaml_write(yaml_dict=yaml_dict, yaml_file=self.work_root + "/final_config.yaml")
