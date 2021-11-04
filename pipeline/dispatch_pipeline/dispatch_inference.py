"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
inference pipeline.
"""
from __future__ import annotations

from os.path import join

import pandas as pd

from pipeline.local_pipeline.core_chain import CoreRoute
from pipeline.local_pipeline.mapping import EnvironmentConfigure
from pipeline.local_pipeline.preprocess_chain import PreprocessRoute
from utils.bunch import Bunch
from utils.constant_values import ConstantValues


class Inference:
    """
    Inference object
    """

    def __init__(self, **params):
        self.name = params[ConstantValues.name]
        self.model_name = params[ConstantValues.model_name]
        self.work_root = params[ConstantValues.work_root]

        self.task_name = params[ConstantValues.task_name]
        self.infer_result_type = params[ConstantValues.infer_result_type]
        self.metric_name = params[ConstantValues.metric_name]
        self.feature_configure_name = params[ConstantValues.feature_configure_name]
        self.data_file_type = params[ConstantValues.data_file_type]
        self.inference_column_name_flag = params[ConstantValues.inference_column_name_flag]
        self.inference_data_path = params[ConstantValues.inference_data_path]

        self.dataset_name = params[ConstantValues.dataset_name]
        self.target_names = params[ConstantValues.target_names]

        self.type_inference_name = params[ConstantValues.type_inference_name]
        self.data_clear_name = params[ConstantValues.data_clear_name]
        self.label_encoder_name = params["label_encoder_name"]
        self.feature_generator_name = params["feature_generator_name"]
        self.unsupervised_feature_selector_name = params["unsupervised_feature_selector_name"]
        self.supervised_feature_selector_name = params["supervised_feature_selector_name"]

        self.data_clear_flag = params["data_clear_flag"]
        self.label_encoder_flag = params["label_encoder_flag"]
        self.feature_generator_flag = params["feature_generator_flag"]
        self.unsupervised_feature_selector_flag = params["unsupervised_feature_selector_flag"]
        self.supervised_feature_selector_flag = params["supervised_feature_selector_flag"]

        self.final_file_path = params[self.model_name]["final_file_path"]
        self.work_model_root = params[self.model_name]["work_model_root"]
        self.increment_flag = params[self.model_name]["increment_flag"]
        assert isinstance(self.increment_flag, bool)

        self.out_put_path = params[ConstantValues.out_put_path]

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
        feature_dict = {"user_feature_path": join(work_feature_root,
                                                  feature_dict.user_feature),

                        "type_inference_feature_path": join(
                            work_feature_root,
                            feature_dict.type_inference_feature),

                        "data_clear_feature_path": join(
                            work_feature_root,
                            feature_dict.data_clear_feature),

                        "feature_generator_feature_path": join(
                            work_feature_root,
                            feature_dict.feature_generator_feature),

                        "unsupervised_feature_path": join(
                            work_feature_root,
                            feature_dict.unsupervised_feature),

                        "supervised_feature_path": join(
                            work_feature_root,
                            feature_dict.supervised_feature),

                        "label_encoding_models_path": join(
                            work_feature_root,
                            feature_dict.label_encoding_path),

                        "impute_models_path": join(
                            work_feature_root,
                            feature_dict.impute_path),

                        "label_encoder_feature_path": join(
                            work_feature_root,
                            feature_dict.label_encoder_feature)
                        }
        preprocessing_params = Bunch(
            name="PreprocessRoute",
            feature_path_dict=feature_dict,
            task_name=self.task_name,
            train_flag=ConstantValues.inference,
            train_data_path=None,
            val_data_path=None,
            inference_data_path=self.inference_data_path,
            inference_column_name_flag=self.inference_column_name_flag,
            data_file_type=self.data_file_type,
            dataset_name=self.dataset_name,
            type_inference_name=self.type_inference_name,
            data_clear_name=self.data_clear_name,
            data_clear_flag=self.data_clear_flag,
            label_encoder_name=self.label_encoder_name,
            label_encoder_flag=self.label_encoder_flag,
            feature_generator_name=self.feature_generator_name,
            feature_generator_flag=self.feature_generator_flag,
            unsupervised_feature_selector_name=self.unsupervised_feature_selector_name,
            unsupervised_feature_selector_flag=self.unsupervised_feature_selector_flag
        )
        preprocess_chain = PreprocessRoute(**preprocessing_params)

        entity_dict = preprocess_chain.run()

        assert ConstantValues.infer_dataset in entity_dict

        core_params = Bunch(
            name="core_route",
            train_flag=ConstantValues.inference,
            task_name=self.task_name,
            model_root_path=self.work_model_root,
            target_feature_configure_path=self.final_file_path,
            pre_feature_configure_path=None,
            infer_result_type=self.infer_result_type,
            model_name=self.model_name,
            increment_flag=self.increment_flag,
            feature_configure_name=self.feature_configure_name,
            label_encoding_path=feature_dict[ConstantValues.label_encoding_models_path],
            metrics_name=self.metric_name,
            supervised_feature_selector_flag=self.supervised_feature_selector_flag
        )
        core_chain = CoreRoute(**core_params)

        core_chain.run(**entity_dict)
        predict_result = core_chain.result
        self.output_result(predict_result=predict_result)
