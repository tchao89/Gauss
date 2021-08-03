# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

from pipeline.core_chain import CoreRoute
from pipeline.preprocess_chain import PreprocessRoute
from pipeline.base import check_data

from gauss_factory.gauss_factory_producer import GaussFactoryProducer

from utils.common_component import yaml_write
from utils.exception import PipeLineLogicError
from pipeline.mapping import EnvironmentConfigure


# pipeline defined by user.
class UdfModelingTree(object):
    def __init__(self,
                 name: str,
                 work_root: str,
                 task_type: str,
                 metric_name: str,
                 train_data_path: str,
                 val_data_path: str = None,
                 target_names=None,
                 feature_configure_path: str = None,
                 dataset_type: str = "plain",
                 type_inference: str = "plain",
                 data_clear: str = "plain",
                 data_clear_flag=None,
                 feature_generator: str = "featuretools",
                 feature_generator_flag=None,
                 unsupervised_feature_selector: str = "unsupervised",
                 unsupervised_feature_selector_flag=None,
                 supervised_feature_selector: str = "supervised",
                 supervised_feature_selector_flag=None,
                 model_zoo=None,
                 auto_ml: str = "plain"
                 ):
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

        if model_zoo is None:
            model_zoo = ["xgboost", "lightgbm", "catboost", "lr_lightgbm", "dnn"]

        if supervised_feature_selector_flag is None:
            supervised_feature_selector_flag = [True, False]

        if unsupervised_feature_selector_flag is None:
            unsupervised_feature_selector_flag = [True, False]

        if feature_generator_flag is None:
            feature_generator_flag = [True, False]

        if data_clear_flag is None:
            data_clear_flag = [True, False]

        self.name = name
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
        self.data_clear_flag = data_clear_flag
        self.feature_generator = feature_generator
        self.feature_generator_flag = feature_generator_flag
        self.unsupervised_feature_selector = unsupervised_feature_selector
        self.unsupervised_feature_selector_flag = unsupervised_feature_selector_flag
        self.supervised_feature_selector = supervised_feature_selector
        self.supervised_feature_selector_flag = supervised_feature_selector_flag
        self.model_zoo = model_zoo
        self.auto_ml = auto_ml
        self.already_data_clear = None
        self.best_model = None
        self.best_metric = None
        self.best_result_root = None
        self.best_model_name = None

        self.pipeline_config = None

    def run_route(self,
                  folder_prefix_str,
                  data_clear_flag,
                  feature_generator_flag,
                  unsupervised_feature_selector_flag,
                  supervised_feature_selector_flag,
                  model_name,
                  auto_ml_path,
                  selector_config_path):

        work_root = self.work_root + "/" + folder_prefix_str

        pipeline_configure = {"data_clear_flag": data_clear_flag,
                              "feature_generator_flag": feature_generator_flag,
                              "unsupervised_feature_selector_flag": unsupervised_feature_selector_flag,
                              "supervised_feature_selector_flag": supervised_feature_selector_flag,
                              "metric_name": self.metric_name,
                              "task_type": self.task_type
                              }

        work_feature_root = work_root + "/feature"

        feature_dict = {"user_feature": self.feature_configure_path,
                        "type_inference_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().type_inference_feature,
                        "data_clear_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().data_clear_feature,
                        "feature_generator_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().feature_generator_feature,
                        "unsupervised_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().unsupervised_feature,
                        "supervised_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().supervised_feature,
                        "label_encoding_path": work_feature_root + "/" + EnvironmentConfigure.feature_dict().label_encoding_path,
                        "impute_path": work_feature_root + "/" + EnvironmentConfigure.feature_dict().impute_path,
                        "final_feature_config": work_feature_root + "/" + EnvironmentConfigure.feature_dict().final_feature_config}

        preprocess_chain = PreprocessRoute(name="PreprocessRoute",
                                           feature_path_dict=feature_dict,
                                           task_type=self.task_type,
                                           train_flag=True,
                                           train_data_path=self.train_data_path,
                                           val_data_path=self.val_data_path,
                                           test_data_path=None,
                                           target_names=self.target_names,
                                           type_inference_name="typeinference",
                                           data_clear_name="plaindataclear",
                                           data_clear_flag=data_clear_flag,
                                           feature_generator_name="featuretools",
                                           feature_generator_flag=feature_generator_flag,
                                           feature_selector_name="unsupervised",
                                           feature_selector_flag=unsupervised_feature_selector_flag)

        try:
            preprocess_chain.run()
        except PipeLineLogicError as e:
            print(e)
            return None

        entity_dict = preprocess_chain.entity_dict
        self.already_data_clear = preprocess_chain.already_data_clear

        # 如果未进行数据清洗, 并且模型需要数据清洗, 则返回None.
        if check_data(already_data_clear=self.already_data_clear, model_name=model_name) is not True:
            return None

        assert "dataset" in entity_dict and "val_dataset" in entity_dict

        work_model_root = work_root + "/model/" + model_name + "/"
        model_save_root = work_model_root + "/model_save"
        model_config_root = work_model_root + "/model_config"
        feature_config_root = work_model_root + "/feature_config"

        core_chain = CoreRoute(name="core_route",
                               train_flag=True,
                               model_save_root=model_save_root,
                               model_config_root=model_config_root,
                               feature_config_root=feature_config_root,
                               target_feature_configure_path=feature_dict["final_feature_config"],
                               pre_feature_configure_path=feature_dict["unsupervised_feature"],
                               model_name=model_name,
                               label_encoding_path=feature_dict["label_encoding_path"],
                               model_type="tree_model",
                               metrics_name=self.metric_name,
                               task_type=self.task_type,
                               feature_selector_name="feature_selector",
                               feature_selector_flag=supervised_feature_selector_flag,
                               auto_ml_type="auto_ml",
                               auto_ml_path=auto_ml_path,
                               selector_config_path=selector_config_path)

        core_chain.run(**entity_dict)
        local_metric = core_chain.optimal_metrics
        assert local_metric is not None
        local_model = core_chain.optimal_model
        return local_model, local_metric, work_root, model_name, pipeline_configure

    @classmethod
    def create_component(cls, component_name: str, **params):

        gauss_factory = GaussFactoryProducer()
        component_factory = gauss_factory.get_factory(choice="component")
        return component_factory.get_component(component_name=component_name, **params)

    @classmethod
    def create_entity(cls, entity_name: str, **params):

        gauss_factory = GaussFactoryProducer()
        entity_factory = gauss_factory.get_factory(choice="entity")
        return entity_factory.get_entity(entity_name=entity_name, **params)

    # local_best_model, local_best_metric, local_best_work_root, local_best_model_name
    def update_best(self, *params):
        if self.best_metric is None or self.best_metric.__cmp__(params[1]) < 0:
            self.best_model = params[0]
            self.best_metric = params[1]
            self.best_result_root = params[2]
            self.best_model_name = params[3]
            self.pipeline_config = params[4]

    def run(self):

        for data_clear in self.data_clear_flag:
            for feature_generator in self.feature_generator_flag:
                for unsupervised_feature_sel in self.unsupervised_feature_selector_flag:
                    for supervise_feature_sel in self.supervised_feature_selector_flag:
                        for model in self.model_zoo:
                            prefix = str(data_clear) + "_" + str(feature_generator) + "_" + str(
                                unsupervised_feature_sel) + "_" + str(supervise_feature_sel)
                            local_result = self.run_route(folder_prefix_str=prefix,
                                                          data_clear_flag=data_clear,
                                                          feature_generator_flag=feature_generator,
                                                          unsupervised_feature_selector_flag=unsupervised_feature_sel,
                                                          supervised_feature_selector_flag=supervise_feature_sel,
                                                          model_name=model,
                                                          auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config",
                                                          selector_config_path="/home/liangqian/PycharmProjects/Gauss/configure_files/selector_config")

                            if local_result is not None:
                                self.update_best(*local_result)

        yaml_dict = {"best_root": self.best_result_root,
                     "best_model_name": self.best_model_name,
                     "work_root": self.work_root,
                     "dataset_name": self.dataset_type,
                     "type_inference": self.type_inference,
                     "data_clear": self.data_clear,
                     "feature_generator": self.feature_generator,
                     "unsupervised_feature_selector": self.unsupervised_feature_selector,
                     "supervised_feature_selector": self.supervised_feature_selector,
                     "auto_ml": self.auto_ml,
                     "best_metric": float(self.best_metric.result)}
        yaml_dict.update(self.pipeline_config)
        yaml_write(yaml_dict=yaml_dict, yaml_file=self.work_root + "/pipeline_config.yaml")
