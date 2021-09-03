# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

import os.path

import pandas as pd

from local_pipeline.core_chain import CoreRoute
from local_pipeline.preprocess_chain import PreprocessRoute

from utils.common_component import yaml_read
from utils.bunch import Bunch


class Inference(object):
    def __init__(self,
                 name: str,
                 work_root: str,
                 model_name: str,
                 out_put_path: str):

        self.name = name
        self.model_name = model_name
        self.work_root = work_root
        self.root_conf = self.work_root + "/" + "inference_user_config.yaml"

        self.conf = Bunch(**yaml_read(self.root_conf))

        self.task_type = self.conf.task_type
        self.metric_name = self.conf.metric_name
        self.test_data_path = self.conf.test_data_path

        self.dataset_name = self.conf.dataset_type
        self.type_inference_name = self.conf.type_inference
        self.data_clear_name = self.conf.data_clear
        self.feature_generator_name = self.conf.feature_generator
        self.unsupervised_feature_selector = self.conf.unsupervised_feature_selector
        self.supervised_feature_selector_name = self.conf.supervised_feature_selector

        self.data_clear_flag = self.conf.data_clear_flag
        self.feature_generator_flag = self.conf.feature_generator_flag
        self.unsupervised_feature_selector_flag = self.conf.unsupervised_feature_selector_flag
        self.supervised_feature_selector_flag = self.conf.supervised_feature_selector_flag

        self.out_put_path = out_put_path

    def output_result(self, predict_result: pd.DataFrame):
        assert isinstance(predict_result, pd.DataFrame)
        predict_result.to_csv(os.path.join(self.out_put_path, "result.csv"), index=False)

    def online_run(self, dataframe):
        pass

    def offline_run(self):
        work_feature_root = self.work_root + "/feature"

        feature_dict = {"user_feature": "null",
                        "type_inference_feature": work_feature_root + "/" + "type_inference_feature.yaml",
                        "feature_generator_feature": work_feature_root + "/" + "feature_generator_feature.yaml",
                        "unsupervised_feature": work_feature_root + "/" + "unsupervised_feature.yaml",
                        "supervised_feature": work_feature_root + "/" + "supervised_feature.yaml",
                        "data_clear_feature": work_feature_root + "/" + "data_clear_feature.yaml",
                        "label_encoding_path": work_feature_root + "/" + "label_encoding_models",
                        "impute_path": work_feature_root + "/" + "impute_models",
                        "final_feature_config": self.conf[self.model_name]["final_file_path"]}

        preprocess_chain = PreprocessRoute(name="PreprocessRoute",
                                           feature_path_dict=feature_dict,
                                           task_type=self.task_type,
                                           train_flag=False,
                                           train_data_path=None,
                                           val_data_path=None,
                                           test_data_path=self.test_data_path,
                                           dataset_name=self.dataset_name,
                                           type_inference_name=self.type_inference_name,
                                           data_clear_name=self.data_clear_name,
                                           data_clear_flag=self.data_clear_flag,
                                           feature_generator_name=self.feature_generator_name,
                                           feature_generator_flag=self.feature_generator_flag,
                                           feature_selector_name=self.unsupervised_feature_selector,
                                           feature_selector_flag=self.unsupervised_feature_selector_flag)

        preprocess_chain.run()
        entity_dict = preprocess_chain.entity_dict

        assert "dataset" in entity_dict
        work_model_root = self.conf[self.model_name]["work_model_root"]
        model_save_root = work_model_root + "/model_save"
        model_config_root = work_model_root + "/model_config"
        feature_config_root = work_model_root + "/feature_config"

        core_chain = CoreRoute(name="core_route",
                               train_flag=False,
                               model_save_root=model_save_root,
                               model_config_root=model_config_root,
                               feature_config_root=feature_config_root,
                               target_feature_configure_path=feature_dict["final_feature_config"],
                               pre_feature_configure_path=None,
                               model_name=self.model_name,
                               label_encoding_path=self.work_root + "/feature/label_encoding_models",
                               metrics_name=self.metric_name,
                               task_type=self.task_type,
                               feature_selector_name=self.supervised_feature_selector_name,
                               feature_selector_flag=self.supervised_feature_selector_flag,
                               auto_ml_type="auto_ml"
                               )

        core_chain.run(**entity_dict)
        predict_result = core_chain.result
        self.output_result(predict_result=predict_result)
