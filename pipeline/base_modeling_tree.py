# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

import abc

from gauss_factory.gauss_factory_producer import GaussFactoryProducer
from utils.common_component import yaml_write
from utils.Logger import logger
from utils.exception import NoResultReturnException


# pipeline defined by user.
class BaseModelingTree(object):
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

        self.pipeline_config = None

    @abc.abstractmethod
    def run_route(self, *params):
        pass

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

    # local_best_model, local_best_metric, local_best_work_root, pipeline configure
    def update_best(self, *params):
        best_model = params[0]
        best_metric = params[1]
        best_result_root = params[2]
        pipeline_config = params[3]

        if self.best_metric is None:
            self.best_model = best_model
            self.best_metric = best_metric
            self.best_result_root = best_result_root
            self.pipeline_config = pipeline_config

        if self.best_metric.__cmp__(best_metric) < 0:

            self.best_model = best_model
            self.best_metric = best_metric
            self.best_result_root = best_result_root
            self.pipeline_config = pipeline_config

    def run(self, *args):
        self._run(*args)
        self.set_pipeline_config()

    @abc.abstractmethod
    def _run(self, *params):
        pass

    def set_pipeline_config(self):

        if self.best_model is None:
            raise NoResultReturnException("Best model is None.")

        yaml_dict = {"best_root": self.best_result_root,
                     "best_model_name": self.best_model.name,
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
                     "best_metric": float(self.best_metric.result)}

        yaml_dict.update(self.pipeline_config)
        yaml_write(yaml_dict=yaml_dict, yaml_file=self.work_root + "/pipeline_config.yaml")
