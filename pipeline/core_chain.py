# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: luoqing

from __future__ import annotations

from typing import Any

from entity.dataset.plain_dataset import PlaintextDataset
from utils.bunch import Bunch

from gauss.component import Component
from gauss_factory.gauss_factory_producer import GaussFactoryProducer
from gauss_factory.entity_factory import MetricsFactory

from utils.common_component import yaml_write, yaml_read, feature_list_generator


class CoreRoute(Component):
    def __init__(self,
                 name: str,
                 train_flag: bool,
                 model_name: str,
                 model_save_root: str,
                 target_feature_configure_path: Any(str, None),
                 pre_feature_configure_path: Any(str, None),
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

        self._feature_config_path = pre_feature_configure_path
        self._final_file_path = target_feature_configure_path

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

        if self._train_flag:
            self._best_model = None
            self._best_metrics = None

        if not self._train_flag:
            self._result = None

    def _train_run(self, **entity):
        assert "dataset" in entity
        assert "val_dataset" in entity

        if self._feature_selector_flag:

            self.feature_selector.run(**entity)
            self._best_model = self.feature_selector.optimal_model
            self._best_metrics = self.feature_selector.optimal_metrics

        else:
            train_dataset = entity["dataset"]

            self.metrics.label_name = train_dataset.get_dataset().target_names[0]
            feature_conf = yaml_read(self._feature_config_path)

            entity["model"] = self.model
            entity["metrics"] = self.metrics

            self.auto_ml.run(**entity)

            self._best_model = self.auto_ml.optimal_model
            self._best_metrics = self.auto_ml.optimal_metrics

            self._best_model.model_save()

            yaml_write(yaml_dict=feature_conf, yaml_file=self._final_file_path)

    def _predict_run(self, **entity):
        """
        :param entity:
        :return: This method will return predict result for test dataset.
        """
        assert "dataset" in entity
        assert self._model_save_path is not None
        assert self._train_flag is False

        dataset = entity.get("dataset")

        if self._feature_selector_flag:
            self.feature_selector.run(**entity)
            self._result = self.feature_selector.result

        else:

            feature_conf = yaml_read(self._final_file_path)
            features = feature_list_generator(feature_dict=feature_conf)
            data = dataset.feature_choose(features)

            data_pair = Bunch(data=data, target=None, target_names=None)
            dataset = PlaintextDataset(name="train_data", task_type=self._train_flag, data_pair=data_pair)

            model_params = Bunch(name=self._model_name,
                                 model_path=self._model_save_path,
                                 train_flag=self._train_flag,
                                 task_type=self._task_type)

            model = self.create_entity(entity_name=self._model_name, **model_params)

            self._result = model.predict(dataset)

    @property
    def optimal_model(self):
        assert self._train_flag
        assert self._best_model is not None
        return self._best_model

    @property
    def optimal_metrics(self):
        assert self._train_flag
        assert self._best_metrics is not None
        return self._best_metrics

    @property
    def result(self):
        assert not self._train_flag
        assert self._result is not None
        return self._result

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
        pass

    def get_eval_result(self,  **entity):
        pass
