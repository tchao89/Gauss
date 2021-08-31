# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

from typing import List

from local_pipeline.core_chain import CoreRoute
from local_pipeline.preprocess_chain import PreprocessRoute
from utils.check_dataset import check_data
from local_pipeline.mapping import EnvironmentConfigure
from local_pipeline.base_modeling_graph import BaseModelingGraph
from utils.common_component import yaml_write

from utils.exception import PipeLineLogicError, NoResultReturnException
from utils.Logger import logger


# local_pipeline defined by user.
class UdfModelingGraph(BaseModelingGraph):
    def __init__(self, name: str, work_root: str, task_type: str, metric_name: str, train_data_path: str,
                 val_data_path: str = None, target_names=None, feature_configure_path: str = None,
                 dataset_type: str = "plain", type_inference: str = "plain", data_clear: str = "plain",
                 data_clear_flag=None, feature_generator: str = "featuretools", feature_generator_flag=None,
                 unsupervised_feature_selector: str = "unsupervised", unsupervised_feature_selector_flag=None,
                 supervised_feature_selector: str = "supervised", supervised_feature_selector_flag=None, model_zoo=None,
                 supervised_selector_names=None, auto_ml: str = "plain", opt_model_names: List[str] = None,
                 auto_ml_path: str = None, selector_config_path: str = None):
        """
        :param name:
        :param work_root:
        :param task_type:
        :param metric_name:
        :param train_data_path:
        :param val_data_path:
        :param feature_configure_path:
        :param dataset_type:
        :param type_inference:
        :param data_clear:
        :param data_clear_flag:
        :param feature_generator:
        :param feature_generator_flag:
        :param unsupervised_feature_selector:
        :param unsupervised_feature_selector_flag:
        :param supervised_feature_selector:
        :param supervised_feature_selector_flag:
        :param model_zoo: model name list
        :param auto_ml: auto ml name
        """

        super().__init__(name=name, work_root=work_root, task_type=task_type, metric_name=metric_name,
                         train_data_path=train_data_path, val_data_path=val_data_path, target_names=target_names,
                         feature_configure_path=feature_configure_path, dataset_type=dataset_type,
                         type_inference=type_inference, data_clear=data_clear, feature_generator=feature_generator,
                         unsupervised_feature_selector=unsupervised_feature_selector,
                         supervised_feature_selector=supervised_feature_selector, auto_ml=auto_ml,
                         opt_model_names=opt_model_names, auto_ml_path=auto_ml_path,
                         selector_config_path=selector_config_path)

        if model_zoo is None:
            model_zoo = ["xgboost", "lightgbm", "catboost", "lr_lightgbm", "dnn"]

        if supervised_feature_selector_flag is None:
            supervised_feature_selector_flag = True

        if unsupervised_feature_selector_flag is None:
            unsupervised_feature_selector_flag = True

        if feature_generator_flag is None:
            feature_generator_flag = True

        if data_clear_flag is None:
            data_clear_flag = True

        self.data_clear_flag = data_clear_flag
        self.feature_generator_flag = feature_generator_flag
        self.unsupervised_feature_selector_flag = unsupervised_feature_selector_flag
        self.supervised_feature_selector_flag = supervised_feature_selector_flag
        self.model_zoo = model_zoo

        self.already_data_clear = None
        self.best_model = None
        self.best_metric = None
        self.best_result_root = None
        self.best_model_name = None

        self._model_names = model_zoo
        self._supervised_selector_names = supervised_selector_names
        self._auto_ml_names = opt_model_names
        self.jobs = None

        self.multiprocess_config = dict()

        self.pipeline_configure = {"work_root": self.work_root,
                                   "data_clear_flag": data_clear_flag,
                                   "data_clear": self.data_clear,
                                   "feature_generator_flag": feature_generator_flag,
                                   "feature_generator": self.feature_generator,
                                   "unsupervised_feature_selector_flag": unsupervised_feature_selector_flag,
                                   "unsupervised_feature_selector": self.unsupervised_feature_selector,
                                   "supervised_feature_selector_flag": supervised_feature_selector_flag,
                                   "supervised_feature_selector": self.supervised_feature_selector,
                                   "metric_name": self.metric_name,
                                   "task_type": self.task_type,
                                   "target_names": self.target_names,
                                   "dataset_type": self.dataset_type,
                                   "type_inference": self.type_inference
                                   }

    def run_route(self,
                  data_clear_flag,
                  feature_generator_flag,
                  unsupervised_feature_selector_flag,
                  supervised_feature_selector_flag,
                  model_name,
                  auto_ml_path,
                  selector_config_path):

        assert isinstance(data_clear_flag, bool) and \
               isinstance(feature_generator_flag, bool) and \
               isinstance(unsupervised_feature_selector_flag, bool) and \
               isinstance(supervised_feature_selector_flag, bool)

        work_root = self.work_root

        work_feature_root = work_root + "/feature"

        feature_dict = {"user_feature": self.feature_configure_path,
                        "type_inference_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().type_inference_feature,
                        "data_clear_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().data_clear_feature,
                        "feature_generator_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().feature_generator_feature,
                        "unsupervised_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().unsupervised_feature,
                        "supervised_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().supervised_feature,
                        "label_encoding_path": work_feature_root + "/" + EnvironmentConfigure.feature_dict().label_encoding_path,
                        "impute_path": work_feature_root + "/" + EnvironmentConfigure.feature_dict().impute_path}

        preprocess_chain = PreprocessRoute(name="PreprocessRoute",
                                           feature_path_dict=feature_dict,
                                           task_type=self.task_type,
                                           train_flag=True,
                                           train_data_path=self.train_data_path,
                                           val_data_path=self.val_data_path,
                                           test_data_path=None,
                                           target_names=self.target_names,
                                           type_inference_name=self.type_inference,
                                           data_clear_name=self.data_clear,
                                           data_clear_flag=data_clear_flag,
                                           feature_generator_name=self.feature_generator,
                                           feature_generator_flag=feature_generator_flag,
                                           feature_selector_name=self.unsupervised_feature_selector,
                                           feature_selector_flag=unsupervised_feature_selector_flag)

        try:
            preprocess_chain.run()
        except PipeLineLogicError as e:
            logger.info(e)
            return None

        entity_dict = preprocess_chain.entity_dict
        self.already_data_clear = preprocess_chain.already_data_clear

        # 如果未进行数据清洗, 并且模型需要数据清洗, 则返回None.
        if check_data(already_data_clear=self.already_data_clear, model_name=model_name) is not True:
            return None

        assert "dataset" in entity_dict and "val_dataset" in entity_dict

        work_model_root = work_root + "/model/" + model_name + "/"
        model_save_root = work_model_root + "/model_save"
        model_config_root = work_model_root + "/model_parameters"
        feature_config_root = work_model_root + "/feature_config"
        feature_dict["final_feature_config"] = \
            feature_config_root + "/" + EnvironmentConfigure.feature_dict().final_feature_config
        core_chain = CoreRoute(name="core_route",
                               train_flag=True,
                               model_save_root=model_save_root,
                               model_config_root=model_config_root,
                               feature_config_root=feature_config_root,
                               target_feature_configure_path=feature_dict["final_feature_config"],
                               pre_feature_configure_path=feature_dict["unsupervised_feature"],
                               model_name=model_name,
                               label_encoding_path=feature_dict["label_encoding_path"],
                               metrics_name=self.metric_name,
                               task_type=self.task_type,
                               feature_selector_name=self._supervised_selector_names,
                               feature_selector_flag=supervised_feature_selector_flag,
                               auto_ml_type="auto_ml",
                               auto_ml_path=auto_ml_path,
                               opt_model_names=self._opt_model_names,
                               selector_config_path=selector_config_path)

        core_chain.run(**entity_dict)
        local_metric = core_chain.optimal_metrics
        assert local_metric is not None
        return {"work_model_root": work_model_root,
                "model_name": model_name,
                "metrics_result": local_metric,
                "final_file_path": feature_dict["final_feature_config"]}

    def _run(self):
        train_results = []
        for model in self.model_zoo:
            local_result = self.run_route(data_clear_flag=self.data_clear_flag,
                                          feature_generator_flag=self.feature_generator_flag,
                                          unsupervised_feature_selector_flag=self.unsupervised_feature_selector_flag,
                                          supervised_feature_selector_flag=self.supervised_feature_selector_flag,
                                          model_name=model,
                                          auto_ml_path=self.auto_ml_path,
                                          selector_config_path=self.selector_config_path)

            if local_result is not None:
                train_results.append(local_result)
            self.find_best_result(train_results=train_results)

    def find_best_result(self, train_results):

        best_result = {}

        if len(train_results) == 0:
            raise NoResultReturnException("All subprocesses have failed.")
        logger.info("%d subprocess(es) have return result(s) successfully.", len(train_results))

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

        self.pipeline_configure.update(best_result)

    def set_pipeline_config(self):

        yaml_dict = {}

        if self.pipeline_configure is not None:
            yaml_dict.update(self.pipeline_configure)

        yaml_write(yaml_dict=yaml_dict, yaml_file=self.work_root + "/pipeline_config.yaml")
