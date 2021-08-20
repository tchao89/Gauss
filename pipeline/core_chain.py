# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: luoqing

from __future__ import annotations

from typing import Any, List

from utils.bunch import Bunch

from gauss.component import Component
from gauss_factory.gauss_factory_producer import GaussFactoryProducer
from gauss_factory.entity_factory import MetricsFactory

from utils.common_component import yaml_write, yaml_read


class CoreRoute(Component):
    def __init__(self,
                 name: str,
                 train_flag: bool,
                 model_name: str,
                 model_save_root: str,
                 model_config_root: str,
                 feature_config_root: str,
                 target_feature_configure_path: Any(str, None),
                 pre_feature_configure_path: Any(str, None),
                 label_encoding_path: str,
                 model_type: str,
                 metrics_name: str,
                 task_type: str,
                 feature_selector_name: str,
                 feature_selector_flag: bool,
                 supervised_selector_name: str = None,
                 auto_ml_type: str = "XXX",
                 auto_ml_path: str = "",
                 auto_ml_name: str = None,
                 opt_model_names: List[str] = None,
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

        self.auto_ml_name = auto_ml_name
        self.supervised_selector_name = supervised_selector_name

        self._model_config_root = model_config_root
        self._feature_config_root = feature_config_root
        self._task_type = task_type
        self._metrics_name = metrics_name
        self._auto_ml_path = auto_ml_path
        self._feature_selector_flag = feature_selector_flag

        self._opt_model_names = opt_model_names

        self._feature_config_path = pre_feature_configure_path

        self._final_file_path = target_feature_configure_path

        self._model_type = model_type

        if self._train_flag:
            self._best_metrics = None
            self._best_model = None
        else:
            self._result = None
            self._feature_conf = None

        # create feature configure
        feature_conf_params = Bunch(name="featureconfigure", file_path=None)
        self.feature_conf = self.create_entity(entity_name="featureconfigure", **feature_conf_params)

        # create metrics and set optimize_mode
        metrics_factory = MetricsFactory()
        metrics_params = Bunch(name=self._metrics_name)
        self.metrics = metrics_factory.get_entity(entity_name=self._metrics_name, **metrics_params)
        self._optimize_mode = self.metrics.optimize_mode

        # create model
        model_params = Bunch(name=self._model_name,
                             model_path=self._model_save_path,
                             model_config_root=model_config_root,
                             feature_config_root=feature_config_root,
                             train_flag=self._train_flag,
                             task_type=self._task_type)

        self.model = self.create_entity(entity_name=self._model_name, **model_params)

        if self.auto_ml_name is not None:
            self._opt_model_names = [self.auto_ml_name]

        tuner_params = Bunch(name=self._auto_ml_type,
                             train_flag=self._train_flag,
                             enable=self.enable,
                             opt_model_names=self._opt_model_names,
                             optimize_mode=self._optimize_mode,
                             auto_ml_path=self._auto_ml_path)

        self.auto_ml = self.create_component(component_name="tabularautoml", **tuner_params)

        # auto_ml_path and selector_config_path are fixed configuration files.
        s_params = Bunch(name=self._feature_selector_name,
                         train_flag=self._train_flag,
                         enable=self.enable,
                         metrics_name=self._metrics_name,
                         task_name=task_type,
                         model_config_root=model_config_root,
                         feature_config_root=feature_config_root,
                         feature_config_path=pre_feature_configure_path,
                         final_file_path=target_feature_configure_path,
                         label_encoding_configure_path=label_encoding_path,
                         selector_config_path=selector_config_path,
                         model_name=self._model_name,
                         auto_ml_path=auto_ml_path,
                         model_save_path=self._model_save_path)

        self.feature_selector = self.create_component(component_name="supervisedfeatureselector", **s_params)

    def _train_run(self, **entity):
        assert "dataset" in entity
        assert "val_dataset" in entity

        entity["model"] = self.model
        entity["metrics"] = self.metrics

        if self._feature_selector_flag:
            entity["feature_configure"] = self.feature_conf
            entity["auto_ml"] = self.auto_ml

            self.feature_selector.run(**entity)

        else:
            train_dataset = entity["dataset"]

            self.metrics.label_name = train_dataset.get_dataset().target_names[0]
            feature_conf = yaml_read(self._feature_config_path)

            self.feature_conf.file_path = self._feature_config_path
            self.feature_conf.parse(method="system")
            self.feature_conf.feature_selector(feature_list=None)

            entity["model"].update_feature_conf(feature_conf=self.feature_conf)
            self.auto_ml.run(**entity)

            yaml_write(yaml_dict=feature_conf, yaml_file=self._final_file_path)

        self._best_model = entity["model"]
        # self._best_metrics is a MetricsResult object.
        self._best_metrics = entity["model"].val_metrics
        entity["model"].model_save()

    def _predict_run(self, **entity):
        """
        :param entity:
        :return: This method will return predict result for test dataset.
        """
        assert "dataset" in entity

        entity["model"] = self.model
        assert self._model_save_path is not None
        assert self._train_flag is False
        entity["feature_configure"] = self.feature_conf
        entity["feature_configure"].file_path = self._final_file_path

        entity["feature_configure"].parse(method="system")

        if self._feature_selector_flag:
            self.feature_selector.run(**entity)
            self._result = self.feature_selector.result

        else:
            self._result = self.model.predict(dataset=entity.get("dataset"))

    @property
    def optimal_metrics(self):
        assert self._train_flag
        assert self._best_metrics is not None
        return self._best_metrics

    @property
    def optimal_model(self):
        assert self._train_flag
        assert self._best_model is not None
        return self._best_model

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
