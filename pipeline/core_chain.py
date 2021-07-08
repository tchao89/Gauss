# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: luoqing

from __future__ import annotations

from gauss.component import Component
from gauss_factory.gauss_factory_producer import GaussFactoryProducer


class CoreRoute(Component):
    def __init__(self,
                 name: str,
                 train_flag: bool,
                 model_save_root: str,
                 target_feature_configure_path: str,
                 pre_feature_configure_path: str,
                 model_type: str ,
                 metric_type: str,
                 task_type: str,
                 feature_selector_name: str,
                 feature_selector_flag: bool,
                 auto_ml_type: str = "XXX"
                 ):

       self.model = ModleFactory(model_type, model_configure_root, model_save_root, train_flag, metric_type, task_type, model_save_root...)
       self.auto_ml = AutoMlFactory(auto_ml_type, ...)
       self.feature_selector = FeatureSelectorFactory(feature_selector_name, feature_selector_flag, pre_feature_configure_path, target_feature_configure_path,...)
       super(CoreRoute, self).__init__(
            name=name,
            train_flag=train_flag
        )

    def run(self, **entity):
        if self._train_flag:
            self._train_run(**entity)
        else:
            self._predict_run(**entity)

    def _train_run(self, **entity):
        entity["model"] = self.model
        if feature_selector_flag and feature_selector_name == "XXX":
            self.feature_selector.run(entity)
        else auto_ml.run(entity)

    def _predict_run(self, **entity):
        self.model.Init()
        model.predict(**entity)

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

    def get_train_loss(self, **entity):
        pass

    def get_train_metric(self, **entity):
        pass

    def get_eval_loss(self, **entity):
        pass
    
    def get_eval_metric(self,  **entity):
        pass

    def get_eval_result(self,  **entity):
        pass
