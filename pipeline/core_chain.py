# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: luoqing

from __future__ import annotations

from utils.bunch import Bunch

from gauss.component import Component
from gauss_factory.gauss_factory_producer import GaussFactoryProducer
from gauss_factory.entity_factory import MetricsFactory


class CoreRoute(Component):
    def __init__(self,
                 name: str,
                 train_flag: bool,
                 model_name: str,
                 model_save_root: str,
                 target_feature_configure_path: str,
                 pre_feature_configure_path: str,
                 label_encoding_path: str,
                 model_type: str,
                 metrics_name: str,
                 task_type: str,
                 feature_selector_name: str,
                 feature_selector_flag: bool,
                 auto_ml_type: str = "XXX",
                 auto_ml_path: str = "",
                 selector_config_path: str = ""
                 ):

        super(CoreRoute, self).__init__(
            name=name,
            train_flag=train_flag
        )

        assert task_type in ["classification", "regression"]

        # name of auto ml object
        self._auto_ml_type = auto_ml_type
        # name of feature_selector_name
        self._feature_selector_name = feature_selector_name
        # name of model, which will be used to create entity
        self._model_name = model_name
        # 模型保存根目录
        self._model_save_path = model_save_root
        self._task_type = task_type
        self._metrics_name = metrics_name
        self._auto_ml_path = auto_ml_path
        self._feature_selector_flag = feature_selector_flag

        self._model_type = model_type

        metrics_factory = MetricsFactory()
        metrics_params = Bunch(name=self._metrics_name)
        self.metrics = metrics_factory.get_entity(entity_name=self._metrics_name, **metrics_params)
        self._optimize_mode = self.metrics.optimize_mode

        model_params = Bunch(name=self._model_name,
                             model_path=self._model_save_path,
                             train_flag=self._train_flag,
                             task_type=self._task_type)

        self.model = self.create_entity(entity_name=self._model_name, **model_params)

        tuner_params = Bunch(name=self._auto_ml_type,
                             train_flag=self._train_flag,
                             enable=self.enable,
                             opt_model_names=["tpe", "random_search", "anneal", "evolution"],
                             optimize_mode=self._optimize_mode,
                             auto_ml_path=self._auto_ml_path)

        self.auto_ml = self.create_component(component_name="tabularautoml", **tuner_params)

        # auto_ml_path and selector_config_path are fixed configuration files.
        s_params = Bunch(name=self._feature_selector_name,
                         train_flag=self._train_flag,
                         enable=self.enable,
                         metrics_name=self._metrics_name,
                         task_name=task_type,
                         feature_config_path=pre_feature_configure_path,
                         final_file_path=target_feature_configure_path,
                         label_encoding_configure_path=label_encoding_path,
                         selector_config_path=selector_config_path,
                         model_name=self._model_name,
                         auto_ml_path=auto_ml_path,
                         model_save_path=self._model_save_path)

        self.feature_selector = self.create_component(component_name="supervisedfeatureselector", **s_params)

        self.best_model = None
        self.best_val_metric = None

    def run(self, **entity):
        if self._train_flag:
            return self._train_run(**entity)
        else:
            return self._predict_run(**entity)

    def _train_run(self, **entity):
        assert "dataset" in entity
        assert "val_dataset" in entity

        if self._feature_selector_flag:
            best_model = self.feature_selector.run(**entity)

        else:
            train_dataset = entity["dataset"]
            self.metrics.label_name = train_dataset.get_dataset().target_names[0]

            entity["model"] = self.model
            entity["metrics"] = self.metrics

            best_model = self.auto_ml.run(**entity)
            best_model.model_save()

        assert best_model is not None
        self.best_model = best_model

        return best_model

    def _predict_run(self, **entity):
        """
        :param entity:
        :return: This method will return predict result for test dataset.
        """
        assert "dataset" in entity
        assert self._model_save_path is not None
        assert self._train_flag is False

        if self._feature_selector_flag:
            result = self.feature_selector.run(**entity)

        else:
            dataset = entity.get("dataset")

            model_params = Bunch(name=self._model_name,
                                 model_path=self._model_save_path,
                                 train_flag=self._train_flag,
                                 task_type=self._task_type)

            model = self.create_entity(entity_name=self._model_name, **model_params)

            return model.predict(dataset)

        return result

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
    
    def get_eval_metric(self):
        assert self.best_model is not None
        return self.best_model.val_metrics

    def get_eval_result(self,  **entity):
        pass
