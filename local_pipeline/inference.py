# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
"""
inference pipeline.
"""
from __future__ import annotations

from os.path import join

import pandas as pd

from local_pipeline.core_chain import CoreRoute
from local_pipeline.mapping import EnvironmentConfigure
from local_pipeline.preprocess_chain import PreprocessRoute
from utils.bunch import Bunch


class Inference:
    """
    Inference object
    """
    def __init__(self, **params):
        self.name = params["name"]
        self.model_name = params["model_name"]
        self.work_root = params["work_root"]

        self.task_name = params["task_name"]
        self.metric_name = params["metric_name"]
        self.feature_configure_name = params["feature_configure_name"]
        self.test_data_path = params["test_data_path"]

        self.dataset_name = params["dataset_name"]
        self.target_names = params["target_names"]
        self.type_inference_name = params["type_inference_name"]
        self.data_clear_name = params["data_clear_name"]
        self.feature_generator_name = params["feature_generator_name"]
        self.unsupervised_feature_selector_name = params["unsupervised_feature_selector_name"]
        self.supervised_feature_selector_name = params["supervised_feature_selector_name"]

        self.data_clear_flag = params["data_clear_flag"]
        self.feature_generator_flag = params["feature_generator_flag"]
        self.unsupervised_feature_selector_flag = params["unsupervised_feature_selector_flag"]
        self.supervised_feature_selector_flag = params["supervised_feature_selector_flag"]

        self.final_file_path = params[self.model_name]["final_file_path"]
        self.work_model_root = params[self.model_name]["work_model_root"]
        self.out_put_path = params["out_put_path"]

    def output_result(self, predict_result: pd.DataFrame):
        """
        Write inference result to csv file.
        :param predict_result:
        :return: None
        """
        assert isinstance(predict_result, pd.DataFrame)
        predict_result.to_csv(join(self.out_put_path, "result.csv"), index=False)

    def online_run(self, dataframe):
        """
        online inference
        :param dataframe:
        :return: None
        """

    def offline_run(self):
        """
        offline inference
        :return:
        """
        work_feature_root = self.work_root + "/feature"

        feature_dict = EnvironmentConfigure.feature_dict()
        feature_dict = {"user_feature": "null",
                        "type_inference_feature": join(
                            work_feature_root,
                            feature_dict.type_inference_feature),

                        "data_clear_feature": join(
                            work_feature_root,
                            feature_dict.data_clear_feature),

                        "feature_generator_feature": join(
                            work_feature_root,
                            feature_dict.feature_generator_feature),

                        "unsupervised_feature": join(
                            work_feature_root,
                            feature_dict.unsupervised_feature),

                        "supervised_feature": join(
                            work_feature_root,
                            feature_dict.supervised_feature),

                        "label_encoding_path": join(
                            work_feature_root,
                            feature_dict.label_encoding_path),

                        "impute_path": join(
                            work_feature_root,
                            feature_dict.impute_path)
                        }
        preprocessing_params = Bunch(
            name="PreprocessRoute",
            feature_path_dict=feature_dict,
            task_name=self.task_name,
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
            unsupervised_feature_selector_name=self.unsupervised_feature_selector_name,
            unsupervised_feature_selector_flag=self.unsupervised_feature_selector_flag
        )
        preprocess_chain = PreprocessRoute(**preprocessing_params)

        preprocess_chain.run()
        entity_dict = preprocess_chain.entity_dict

        assert "dataset" in entity_dict
        model_save_root = self.work_model_root + "/model_save"
        model_config_root = self.work_model_root + "/model_config"
        feature_config_root = self.work_model_root + "/feature_config"

        core_params = Bunch(
            name="core_route",
            train_flag=False,
            model_save_root=model_save_root,
            model_config_root=model_config_root,
            feature_config_root=feature_config_root,
            target_feature_configure_path=self.final_file_path,
            pre_feature_configure_path=None,
            model_name=self.model_name,
            feature_configure_name=self.feature_configure_name,
            label_encoding_path=feature_dict["label_encoding_path"],
            metrics_name=self.metric_name,
            task_name=self.task_name,
            supervised_feature_selector_flag=self.supervised_feature_selector_flag
        )
        core_chain = CoreRoute(**core_params)

        core_chain.run(**entity_dict)
        predict_result = core_chain.result
        self.output_result(predict_result=predict_result)
