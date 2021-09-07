# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
"""
This pipeline is used to train model, which parameters and settings can be customized by user.
"""
from __future__ import annotations

from os.path import join

from local_pipeline.core_chain import CoreRoute
from local_pipeline.preprocess_chain import PreprocessRoute
from local_pipeline.mapping import EnvironmentConfigure
from local_pipeline.base_modeling_graph import BaseModelingGraph

from utils.bunch import Bunch
from utils.check_dataset import check_data
from utils.common_component import yaml_write
from utils.exception import PipeLineLogicError, NoResultReturnException
from utils.Logger import logger


# local_pipeline defined by user.
class UdfModelingGraph(BaseModelingGraph):
    """
    UdfModelingGraph object.
    """
    def __init__(self, name: str, **params):
        """

        :param name: string project, pipeline name
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
        :param data_clear_flag:
        :param feature_generator_name:
        :param feature_generator_flag:
        :param unsupervised_feature_selector_name:
        :param unsupervised_feature_selector_flag:
        :param supervised_feature_selector_name:
        :param supervised_feature_selector_flag:
        :param model_zoo:
        :param supervised_selector_model_names:
        :param auto_ml_name:
        :param opt_model_names:
        :param auto_ml_path:
        :param selector_config_path:
        """

        super().__init__(
            name=name,
            work_root=params["work_root"],
            task_name=params["task_name"],
            metric_name=params["metric_name"],
            train_data_path=params["train_data_path"],
            val_data_path=params["val_data_path"],
            target_names=params["target_names"],
            feature_configure_path=params["feature_configure_path"],
            feature_configure_name=params["feature_configure_name"],
            dataset_name=params["dataset_name"],
            type_inference_name=params["type_inference_name"],
            label_encoder_name=params["label_encoder_name"],
            data_clear_name=params["data_clear_name"],
            feature_generator_name=params["feature_generator_name"],
            unsupervised_feature_selector_name=params["unsupervised_feature_selector_name"],
            supervised_feature_selector_name=params["supervised_feature_selector_name"],
            supervised_selector_model_names=params["supervised_selector_model_names"],
            selector_trial_num=params["selector_trial_num"],
            auto_ml_name=params["auto_ml_name"],
            auto_ml_trial_num=params["auto_ml_trial_num"],
            opt_model_names=params["opt_model_names"],
            auto_ml_path=params["auto_ml_path"],
            selector_config_path=params["selector_config_path"]
        )

        if params["model_zoo"] is None:
            params["model_zoo"] = ["xgboost", "lightgbm", "catboost", "lr_lightgbm", "dnn"]

        if params["supervised_feature_selector_flag"] is None:
            params["supervised_feature_selector_flag"] = True

        if params["unsupervised_feature_selector_flag"] is None:
            params["unsupervised_feature_selector_flag"] = True

        if params["feature_generator_flag"] is None:
            params["feature_generator_flag"] = True

        if params["data_clear_flag"] is None:
            params["data_clear_flag"] = True

        self._flag_dict = Bunch(
            data_clear_flag=params["data_clear_flag"],
            label_encoder_flag=params["label_encoder_name"],
            feature_generator_flag=params["feature_generator_flag"],
            unsupervised_feature_selector_flag=params["unsupervised_feature_selector_flag"],
            supervised_feature_selector_flag=params["supervised_feature_selector_flag"]
        )

        self._model_zoo = params["model_zoo"]

        self.__pipeline_configure = \
            {"work_root":
                 self._work_paths["work_root"],
             "data_clear_flag":
                 self._flag_dict["data_clear_flag"],
             "data_clear_name":
                 self._component_names["data_clear_name"],
             "feature_generator_flag":
                 self._flag_dict["feature_generator_flag"],
             "feature_generator_name":
                 self._component_names["feature_generator_name"],
             "unsupervised_feature_selector_flag":
                 self._flag_dict["unsupervised_feature_selector_flag"],
             "unsupervised_feature_selector_name":
                 self._component_names["unsupervised_feature_selector_name"],
             "supervised_feature_selector_flag":
                 self._flag_dict["supervised_feature_selector_flag"],
             "supervised_feature_selector_name":
                 self._component_names["supervised_feature_selector_name"],
             "metric_name":
                 self._entity_names["metric_name"],
             "task_name":
                 self._attributes_names["task_name"],
             "target_names":
                 self._attributes_names["target_names"],
             "dataset_name":
                 self._entity_names["dataset_name"],
             "type_inference_name":
                 self._component_names["type_inference_name"]
             }

    def _run_route(self, **params):

        assert isinstance(self._flag_dict["data_clear_flag"], bool) and \
               isinstance(self._flag_dict["feature_generator_flag"], bool) and \
               isinstance(self._flag_dict["unsupervised_feature_selector_flag"], bool) and \
               isinstance(self._flag_dict["supervised_feature_selector_flag"], bool)

        work_root = self._work_paths["work_root"]

        work_feature_root = work_root + "/feature"
        feature_dict = EnvironmentConfigure.feature_dict()
        feature_dict = \
            {"user_feature": self._work_paths["feature_configure_path"],
             "type_inference_feature": join(work_feature_root,
                                            feature_dict.type_inference_feature),

             "data_clear_feature": join(work_feature_root,
                                        feature_dict.data_clear_feature),

             "feature_generator_feature": join(work_feature_root,
                                               feature_dict.feature_generator_feature),

             "unsupervised_feature": join(work_feature_root,
                                          feature_dict.unsupervised_feature),

             "supervised_feature": join(work_feature_root,
                                        feature_dict.supervised_feature),

             "label_encoding_path": join(work_feature_root,
                                         feature_dict.label_encoding_path),

             "impute_path": join(work_feature_root,
                                 feature_dict.impute_path),

             "label_encoder_feature": join(work_feature_root,
                                           feature_dict.label_encoder_feature)
             }

        preprocess_chain = PreprocessRoute(
            name="PreprocessRoute",
            feature_path_dict=feature_dict,
            task_name=self._attributes_names["task_name"],
            train_flag=True,
            train_data_path=self._work_paths["train_data_path"],
            val_data_path=self._work_paths["val_data_path"],
            test_data_path=None,
            target_names=self._attributes_names["target_names"],
            dataset_name=self._entity_names["dataset_name"],
            type_inference_name=self._component_names["type_inference_name"],
            data_clear_name=self._component_names["data_clear_name"],
            data_clear_flag=self._flag_dict["data_clear_flag"],
            label_encoder_name=self._component_names["label_encoder_name"],
            label_encoder_flag=self._flag_dict["label_encoder_flag"],
            feature_generator_name=self._component_names["feature_generator_name"],
            feature_generator_flag=self._flag_dict["feature_generator_flag"],
            unsupervised_feature_selector_name=self._component_names["unsupervised_feature_selector_name"],
            unsupervised_feature_selector_flag=self._flag_dict["unsupervised_feature_selector_flag"]
        )

        try:
            preprocess_chain.run()
        except PipeLineLogicError as error:
            logger.info(error)
            return None

        entity_dict = preprocess_chain.entity_dict
        self._already_data_clear = preprocess_chain.already_data_clear

        assert params.get("model_name") is not None
        # 如果未进行数据清洗, 并且模型需要数据清洗, 则返回None.
        if check_data(already_data_clear=self._already_data_clear,
                      model_name=params.get("model_name")) is not True:
            return None

        assert "train_dataset" in entity_dict and "val_dataset" in entity_dict

        work_model_root = work_root + "/model/" + params.get("model_name") + "/"
        model_save_root = work_model_root + "/model_save"
        model_config_root = work_model_root + "/model_parameters"
        feature_config_root = work_model_root + "/feature_config"
        feature_dict["final_feature_config"] = \
            feature_config_root + "/" + EnvironmentConfigure.feature_dict().final_feature_config

        core_chain = CoreRoute(
            name="core_route",
            train_flag=True,
            model_save_root=model_save_root,
            model_config_root=model_config_root,
            feature_config_root=feature_config_root,
            target_feature_configure_path=feature_dict["final_feature_config"],
            pre_feature_configure_path=feature_dict["unsupervised_feature"],
            model_name=params.get("model_name"),
            feature_configure_name=self._entity_names["feature_configure_name"],
            label_encoding_path=feature_dict["label_encoding_path"],
            metrics_name=self._entity_names["metric_name"],
            task_name=self._attributes_names["task_name"],
            supervised_selector_name=self._component_names["supervised_feature_selector_name"],
            feature_selector_model_names=self._global_values["supervised_selector_model_names"],
            selector_trial_num=self._global_values["selector_trial_num"],
            supervised_feature_selector_flag=self._flag_dict["supervised_feature_selector_flag"],
            auto_ml_name=self._component_names["auto_ml_name"],
            auto_ml_trial_num=self._global_values["auto_ml_trial_num"],
            auto_ml_path=self._work_paths["auto_ml_path"],
            opt_model_names=self._global_values["opt_model_names"],
            selector_config_path=self._work_paths["selector_config_path"]
        )

        core_chain.run(**entity_dict)
        local_metric = core_chain.optimal_metrics
        assert local_metric is not None
        return {"work_model_root": work_model_root,
                "model_name": params.get("model_name"),
                "metrics_result": local_metric,
                "final_file_path": feature_dict["final_feature_config"]}

    def _run(self):
        train_results = []

        for model in self._model_zoo:
            local_result = self._run_route(model_name=model)

            if local_result is not None:
                train_results.append(local_result)
            self._find_best_result(train_results=train_results)

    def _find_best_result(self, train_results):

        best_result = {}

        if len(train_results) == 0:
            raise NoResultReturnException("All subprocesses have failed.")

        for result in train_results:
            model_name = result.get("model_name")

            if best_result.get(model_name) is None:
                best_result[model_name] = result

            else:
                if result.get("metrics_result") is not None:
                    if best_result.get(model_name).get("metrics_result").__cmp__(
                            result.get("metrics_result")) < 0:
                        best_result[model_name] = result

        for result in train_results:
            result["metrics_result"] = float(result.get("metrics_result").result)

        self.__pipeline_configure.update(best_result)

    def _set_pipeline_config(self):
        yaml_dict = {}

        if self.__pipeline_configure is not None:
            yaml_dict.update(self.__pipeline_configure)

        yaml_write(yaml_dict=yaml_dict,
                   yaml_file=self._work_paths["work_root"] + "/pipeline_config.yaml")

    @property
    def pipeline_configure(self):
        """
        property method
        :return: A dict of udf model graph configuration.
        """
        if self.__pipeline_configure is None:
            raise RuntimeError("This pipeline has not start.")
        return self.__pipeline_configure
