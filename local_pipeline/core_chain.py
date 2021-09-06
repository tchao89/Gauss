# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: luoqing
"""
Model training pipeline.
"""
from __future__ import annotations

from gauss.component import Component
from gauss_factory.gauss_factory_producer import GaussFactoryProducer
from gauss_factory.entity_factory import MetricsFactory

from utils.common_component import yaml_write, yaml_read
from utils.Logger import logger
from utils.base import get_current_memory_gb
from utils.bunch import Bunch


class CoreRoute(Component):
    """
    CoreRoute object.
    """

    def __init__(self, **params):

        super().__init__(
            name=params["name"],
            train_flag=params["train_flag"]
        )

        assert params["task_name"] in ["classification", "regression"]

        # name of feature_selector_name
        self._feature_selector_name = params.get("supervised_selector_name")
        # name of model, which will be used to create entity
        self._model_name = params["model_name"]
        # 模型保存根目录
        self._model_save_path = params["model_save_root"]

        self.auto_ml_name = params.get("auto_ml_name")

        self._model_config_root = params["model_config_root"]
        self._feature_config_root = params["feature_config_root"]
        self._task_type = params["task_name"]
        self._metrics_name = params["metrics_name"]
        self._feature_selector_flag = params["supervised_feature_selector_flag"]

        self._opt_model_names = params.get("opt_model_names")

        self._feature_config_path = params["pre_feature_configure_path"]

        self._final_file_path = params["target_feature_configure_path"]

        # create feature configure
        feature_conf_params = Bunch(name=["feature_configure_name"], file_path=None)
        self.feature_conf = self.create_entity(
            entity_name=params["feature_configure_name"],
            **feature_conf_params
        )

        # create metrics and set optimize_mode
        metrics_factory = MetricsFactory()
        metrics_params = Bunch(name=self._metrics_name)
        self.metrics = metrics_factory.get_entity(
            entity_name=self._metrics_name,
            **metrics_params
        )
        self._optimize_mode = self.metrics.optimize_mode

        # create model
        model_params = Bunch(
            name=self._model_name,
            model_path=self._model_save_path,
            model_config_root=params["model_config_root"],
            feature_config_root=params["feature_config_root"],
            train_flag=self._train_flag,
            task_type=self._task_type
        )

        self.model = self.create_entity(entity_name=self._model_name, **model_params)

        if self._train_flag:
            self._auto_ml_path = params["auto_ml_path"]
            self.selector_config_path = params["selector_config_path"]

            tuner_params = Bunch(
                name=self.auto_ml_name,
                train_flag=self._train_flag,
                enable=self.enable,
                auto_ml_trial_num=params["auto_ml_trial_num"],
                opt_model_names=self._opt_model_names,
                optimize_mode=self._optimize_mode,
                auto_ml_path=self._auto_ml_path
            )

            self.auto_ml = self.create_component(
                component_name=params["auto_ml_name"],
                **tuner_params
            )
            # auto_ml_path and selector_config_path are fixed configuration files.
            s_params = Bunch(
                name=self._feature_selector_name,
                train_flag=self._train_flag,
                enable=self.enable,
                metrics_name=self._metrics_name,
                task_name=params["task_name"],
                model_config_root=params["model_config_root"],
                feature_config_root=params["feature_config_root"],
                feature_config_path=params["pre_feature_configure_path"],
                final_file_path=params["target_feature_configure_path"],
                label_encoding_configure_path=params["label_encoding_path"],
                feature_selector_model_names=params["feature_selector_model_names"],
                selector_trial_num=params["selector_trial_num"],
                selector_config_path=self.selector_config_path,
                model_name=self._model_name,
                auto_ml_path=params["auto_ml_path"],
                model_save_path=self._model_save_path
            )

            self.feature_selector = self.create_component(
                component_name=params["supervised_selector_name"],
                **s_params
            )
        else:
            self._result = None
            self._feature_conf = None

    def _train_run(self, **entity):

        assert "dataset" in entity
        assert "val_dataset" in entity

        entity["model"] = self.model
        entity["metrics"] = self.metrics

        if self._feature_selector_flag is True:
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

            logger.info(
                "Auto machine learning component has started, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            self.auto_ml.run(**entity)

            yaml_write(yaml_dict=feature_conf, yaml_file=self._final_file_path)

        self._best_model = entity["model"]
        # self._best_metrics is a MetricsResult object.
        self._best_metrics = entity["model"].val_best_metric_result
        print(entity["model"].metrics_history)
        print(max(entity["model"].metrics_history))
        print(len(entity["model"].metrics_history))
        entity["model"].model_save()

    def _predict_run(self, **entity):
        """
        :param entity:
        :return: This method will return predict result for test dataset.
        """
        assert "dataset" in entity
        assert self._model_save_path is not None
        assert self._train_flag is False

        if self._feature_selector_flag:
            assert self.train_flag is False
            entity["feature_configure"] = self.feature_conf
            entity["feature_configure"].file_path = self._final_file_path
            entity["feature_configure"].parse(method="system")

            dataset = entity["dataset"]
            feature_config = entity["feature_configure"]

            assert self._model_save_path
            assert self._final_file_path

            self._result = self.model.predict(dataset=dataset, feature_conf=feature_config)

        else:
            self._result = self.model.predict(dataset=entity.get("dataset"))

    @property
    def optimal_metrics(self):
        """
        Best metric for this graph
        :return: MetricResult
        """
        assert self._train_flag
        assert self._best_metrics is not None
        return self._best_metrics

    @property
    def result(self):
        """
        :return: inference result, pd.Dataframe
        """
        assert not self._train_flag
        assert self._result is not None
        return self._result

    @classmethod
    def create_component(cls, component_name: str, **params):
        """
        :param component_name:
        :param params:
        :return: component object
        """
        gauss_factory = GaussFactoryProducer()
        component_factory = gauss_factory.get_factory(choice="component")
        return component_factory.get_component(component_name=component_name, **params)

    @classmethod
    def create_entity(cls, entity_name: str, **params):
        """

        :param entity_name:
        :param params:
        :return: entity object
        """
        gauss_factory = GaussFactoryProducer()
        entity_factory = gauss_factory.get_factory(choice="entity")
        return entity_factory.get_entity(entity_name=entity_name, **params)
