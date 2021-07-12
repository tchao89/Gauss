# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

import os

from pipeline.core_chain import CoreRoute
from pipeline.preprocess_chain import PreprocessRoute

from typing import List
from utils.common_component import yaml_write
from utils.bunch import Bunch


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
        self.need_data_clear = False
        self.best_model = None
        self.best_metric = None
        self.best_result_root = None

    def run_route(self,
                  folder_prefix_str,
                  data_clear_flag: bool,
                  feature_generator_flag: bool,
                  unsupervised_feature_generator_flag: bool,
                  supervised_feature_selector_flag: bool,
                  model_zoo: List[str]):

        work_root = self.work_root + "/" + folder_prefix_str
        pipeline_configure_path = work_root + "/" + "pipeline.configure"
        pipeline_configure = {"data_clear_flag": data_clear_flag, "feature_generator_flag": feature_generator_flag,
                              "metric_name": self.metric_name, "task_type": self.task_type}
        yaml_write(yaml_file=pipeline_configure_path, yaml_dict=pipeline_configure)

        work_feature_root = work_root + "/feature"
        feature_dict = Bunch()
        feature_dict.user_feature = self.feature_configure_path
        feature_dict.type_inference_feature = os.path.join(work_feature_root, "type_inference_feature.yaml")
        feature_dict.data_clear_feature = os.path.join(work_feature_root, "data_clear_feature.yaml")
        feature_dict.feature_generator_feature = os.path.join(work_feature_root, "feature_generator_feature.yaml")
        feature_dict.unsupervised_feature = os.path.join(work_feature_root, "unsupervised_feature.yaml")
        feature_dict.label_encoding_path = os.path.join(work_feature_root, "label_encoding_path")
        feature_dict.supervised_feature = os.path.join(work_feature_root, "target_feature_feature.yaml")

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

        entity_dict = preprocess_chain.run()
        self.need_data_clear = preprocess_chain.need_data_clear
        assert "dataset" in entity_dict and "val_dataset" in entity_dict

        best_model = None
        best_metric = None
        model_name = None

        for model in model_zoo:
            work_model_root = work_root + "/model/" + model + "/"
            model_save_root = work_model_root + "/model_save"
            # model_config_root = work_model_root + "/model_config"

            core_chain = CoreRoute(name="core_route",
                                   train_flag=True,
                                   model_name=model,
                                   model_save_root=model_save_root,
                                   target_feature_configure_path=feature_dict.supervised_feature,
                                   pre_feature_configure_path=feature_dict.unsupervised_feature,
                                   label_encoding_path=feature_dict.label_encoding_path,
                                   model_type="tree_model",
                                   metrics_name=self.metric_name,
                                   task_type=self.task_type,
                                   feature_selector_name="supervised_selector",
                                   feature_selector_flag=supervised_feature_selector_flag,
                                   auto_ml_type="auto_ml",
                                   auto_ml_path="",
                                   selector_config_path="")

            local_model = core_chain.run(**entity_dict)
            local_metric = best_model.get_val_metric()

            if best_model is None:
                best_model = local_model
            if self.best_metric is None:
                best_metric = local_metric

            if (self.compare(local_metric, best_metric)) < 0:
                best_model = local_model
                best_metric = local_metric
                model_name = model_name

        return best_model, best_metric, work_root, model_name

    @classmethod
    def compare(cls, local_best_metric, best_metric):
        return best_metric - local_best_metric

    # local_best_model, local_best_metric, local_best_work_root, local_best_model_name
    def update_best(self, *params):
        if params[0] is None or self.compare(params[1], self.best_metric) < 0:
            self.best_model = params[0]
            self.best_metric = params[1]
            self.best_result_root = params[2]

    def run(self):
        self.update_best(self.run_route(
            folder_prefix_str="no-clear_no-feagen_no-unsupfeasel_no-supfeasel",
            data_clear_flag=False,
            feature_generator_flag=False,
            unsupervised_feature_generator_flag=False,
            supervised_feature_selector_flag=False,
            model_zoo=["xgboost, lightgbm, catboost"]))

        self.update_best(
            self.run_route("clear_feagen_supfeasel_no-supfeasel", True, True, True, False, ["xgb, lightgbm, catboost"]))

        self.update_best(
            self.run_route("no-clear_no-feagen_no-feasel", True, True, True, True, ["lr+lightgbm_V1", "lr+lightgbm_V2"]))
