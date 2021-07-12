# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

import pandas as pd

from pipeline.core_chain import CoreRoute
from pipeline.preprocess_chain import PreprocessRoute

from utils.common_component import yaml_read


class Inference(object):
    def __init__(self,
                 name: str,
                 work_root: str,
                 out_put_path: str):

        self.name = name
        self.work_root = work_root
        self.root_conf = self.work_root + "/" + "pipeline.configure"
        self.conf = yaml_read(self.root_conf)
        self.task_type = self.conf.task_type
        self.metric_name = self.conf.metric_name
        self.test_data_path = self.test_data_path
        self.target_names = self.conf.target_names
        self.data_clear_flag = self.conf.data_clear_flag
        self.feature_generator_flag = self.conf.feature_generator_flag
        self.unsupervised_feature_name = self.conf.unsupervised_feature_name
        self.supervised_feature_selector_flag = self.conf.supervised_feature_selector_flag
        self.dataset_name = self.conf.dataset_name
        self.type_inference_name = self.conf.type_inference_name
        self.data_clear_name = self.conf.data_clear_name
        self.feature_generator_name = self.conf.feature_generator_name
        self.unsupervised_feature_selector = self.conf.unsupervised_feature_selector
        self.supervised_feature_selector_name = self.conf.supervised_feature_selector_name
        self.data_clear_flag = self.conf.data_clear_flag
        self.feature_generator_flag = self.conf.feature_generator_flag
        self.unsupervised_feature_selector_flag = self.conf.unsupervised_feature_selector_flag
        self.supervised_feature_selector_flag = self.conf.supervised_feature_selector_flag

        self.out_put_path = out_put_path

    def output_result(self, predict_result: pd.DataFrame):
        predict_result.to_csv(self.out_put_path)

    def run(self):
        work_feature_root = self.work_root + "/feature"
        feature_dict = {"user_feature": "null",
                        "type_inference_feature": work_feature_root + "/." + "type_inference_feature",
                        "feature_generator_feature": work_feature_root + "/." + "feature_generate",
                        "unsupervised_feature": work_feature_root + "/." + "unsupervised_feature_selector",
                        "supervised_feature": work_feature_root + "/." + "supervise_feature_selector"}

        preprocess_chain = PreprocessRoute(name="PreprocessRoute",
                                           feature_path_dict=feature_dict,
                                           task_type=self.task_type,
                                           train_flag=False,
                                           train_data_path=None,
                                           val_data_path=None,
                                           test_data_path=self.test_data_path,
                                           target_names=self.target_names,
                                           dataset_name=self.dataset_name,
                                           type_inference_name=self.type_inference_name,
                                           data_clear_name=self.data_clear_name,
                                           data_clear_flag=self.data_clear_flag,
                                           feature_generator_name=self.feature_generator_name,
                                           feature_generator_flag=self.feature_generator_flag,
                                           feature_selector_name=self.unsupervised_feature_selector,
                                           feature_selector_flag=self.unsupervised_feature_selector_flag)

        entity_dict = preprocess_chain.run()

        assert "dataset" in entity_dict
        work_model_root = self.work_root + "/model/" + self.conf.model + "/"
        model_save_root = work_model_root + "/model_save"
        model_config_root = work_model_root + "/model_config"
        model_conf = yaml_read(model_config_root)

        core_chain = CoreRoute(name="core_route",
                               train_flag=False,
                               model_save_root=model_save_root,
                               target_feature_configure_path=feature_dict["supervised_feature"],
                               pre_feature_configure_path=feature_dict["unsupervised_feature"],
                               model_name=model_conf.model_name,
                               label_encoding_path=self.conf.label_encoding_path,
                               model_type=model_conf.model_type,
                               metrics_name=self.metric_name,
                               task_type=self.task_type,
                               feature_selector_name=self.supervised_feature_selector_name,
                               feature_selector_flag=self.supervised_feature_selector_flag,
                               auto_ml_type="auto_ml",
                               auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config",
                               selector_config_path="/home/liangqian/PycharmProjects/Gauss/configure_files/selector_config"
                               )

        predict_result = core_chain.run(**entity_dict)
        self.output_result(predict_result=predict_result)
