"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-Lab
Model training pipeline.
"""
from __future__ import annotations

from gauss.component import Component
from gauss_factory.gauss_factory_producer import GaussFactoryProducer
from utils.constant_values import ConstantValues

from utils.yaml_exec import yaml_read
from utils.yaml_exec import yaml_write
from utils.Logger import logger
from utils.base import get_current_memory_gb
from utils.bunch import Bunch


class CoreRoute(Component):
    """
    CoreRoute object.
    """
    def __init__(self, **params):
        super().__init__(
            name=params[ConstantValues.name],
            train_flag=params[ConstantValues.train_flag],
            task_name=params[ConstantValues.task_name]
        )

        assert params[ConstantValues.task_name] in [ConstantValues.binary_classification,
                                                    ConstantValues.multiclass_classification,
                                                    ConstantValues.regression]

        # name of model, which will be used to create entity
        self.__model_name = params[ConstantValues.model_name]
        self.__feature_selector_flag = params[ConstantValues.supervised_feature_selector_flag]
        self.__feature_configure_path = params[ConstantValues.pre_feature_configure_path]
        self.__final_file_path = params[ConstantValues.target_feature_configure_path]

        # create feature configure
        feature_conf_params = Bunch(name=[ConstantValues.feature_configure_name],
                                    file_path=None)
        self.__feature_conf = self.create_entity(
            entity_name=params[ConstantValues.feature_configure_name],
            **feature_conf_params
        )

        self.__best_metric = None
        model_params = Bunch(
            name=self.__model_name,
            model_root_path=params[ConstantValues.model_root_path],
            train_flag=self._train_flag,
            task_name=self._task_name
        )

        if self._train_flag == ConstantValues.train:
            self.__auto_ml_name = params[ConstantValues.auto_ml_name]
            self.__auto_ml_path = params[ConstantValues.auto_ml_path]
            self.__opt_model_names = params[ConstantValues.opt_model_names]
            self.__selector_configure_path = params[ConstantValues.selector_configure_path]

            model_params.init_model_root = params[ConstantValues.init_model_root]
            model_params.metric_eval_used_flag = params[ConstantValues.metric_eval_used_flag]

            self.__loss_name = params[ConstantValues.loss_name]
            if self.__loss_name is not None:
                loss_params = Bunch(
                    name=self.__loss_name
                )
                self.__loss = self.create_entity(
                    entity_name=self.__loss_name,
                    **loss_params
                )
            else:
                self.__loss = None

            self.__metric_name = params[ConstantValues.metric_name]
            # create metric and set optimize_mode
            metric_params = Bunch(name=self.__metric_name)
            self.__metric = self.create_entity(
                entity_name=self.__metric_name,
                **metric_params
            )
            self.__optimize_mode = self.__metric.optimize_mode

            tuner_params = Bunch(
                name=self.__auto_ml_name,
                train_flag=self._train_flag,
                enable=self.enable,
                task_name=params[ConstantValues.task_name],
                auto_ml_trial_num=params[ConstantValues.auto_ml_trial_num],
                opt_model_names=self.__opt_model_names,
                optimize_mode=self.__optimize_mode,
                auto_ml_path=self.__auto_ml_path
            )

            self.__auto_ml = self.create_component(
                component_name=params[ConstantValues.auto_ml_name],
                **tuner_params
            )

            if self.__feature_selector_flag is True:
                assert params["supervised_selector_mode"] in ['model_select', 'topk_select']
                self.__supervised_selector_mode = params["supervised_selector_mode"]
                if params["supervised_selector_mode"] == "model_select":
                    # auto_ml_path and selector_configure_path are fixed configuration files.
                    s_params = Bunch(
                        name=params["supervised_selector_name"],
                        train_flag=self._train_flag,
                        enable=self._enable,
                        metric_name=self.__metric_name,
                        task_name=params["task_name"],
                        model_root_path=params["model_root_path"],
                        feature_configure_path=params["pre_feature_configure_path"],
                        final_file_path=params["target_feature_configure_path"],
                        feature_selector_model_names=params["feature_selector_model_names"],
                        selector_trial_num=params["selector_trial_num"],
                        selector_configure_path=self.__selector_configure_path,
                        model_name=self.__model_name,
                        auto_ml_path=params["auto_ml_path"],
                    )

                    self.__feature_selector = self.create_component(
                        component_name=params["supervised_selector_name"],
                        **s_params
                    )
                else:
                    # auto_ml_path and selector_configure_path are fixed configuration files.
                    s_params = Bunch(
                        name=params["supervised_selector_name"],
                        train_flag=self._train_flag,
                        enable=self._enable,
                        metric_name=self.__metric_name,
                        task_name=params["task_name"],
                        model_root_path=params["model_root_path"],
                        feature_configure_path=params["pre_feature_configure_path"],
                        final_file_path=params["target_feature_configure_path"],
                        feature_selector_model_names=params["feature_selector_model_names"],
                        improved_selector_configure_path=params["improved_selector_configure_path"],
                        feature_model_trial=params["feature_model_trial"],
                        selector_trial_num=params["selector_trial_num"],
                        selector_configure_path=self.__selector_configure_path,
                        model_name=self.__model_name,
                        auto_ml_path=params["auto_ml_path"],
                    )

                    self.__feature_selector = self.create_component(
                        component_name=params["improved_supervised_selector_name"],
                        **s_params
                    )

                    # This object is used in improved_supervised_selector
                    selector_model_params = Bunch(
                        name="lightgbm",
                        model_root_path=params["model_root_path"],
                        init_model_root=None,
                        train_flag=self._train_flag,
                        task_name=self._task_name,
                        metric_eval_used_flag=params["metric_eval_used_flag"]
                    )

                    model_params.init_model_root = params["init_model_root"]
                    self.__selector_model = self.create_entity(entity_name="lightgbm", **selector_model_params)

                    selector_metric_params = Bunch(name=self.__metric_name)
                    self.__selector_metric = self.create_entity(
                        entity_name=self.__metric_name,
                        **selector_metric_params
                    )

                    selector_tuner_params = Bunch(
                        name=self.__auto_ml_name,
                        train_flag=self._train_flag,
                        enable=self._enable,
                        task_name=params[ConstantValues.task_name],
                        auto_ml_trial_num=params[ConstantValues.auto_ml_trial_num],
                        opt_model_names=self.__opt_model_names,
                        optimize_mode=self.__optimize_mode,
                        auto_ml_path=self.__auto_ml_path
                    )

                    self.__selector_auto_ml = self.create_component(
                        component_name=params[ConstantValues.auto_ml_name],
                        **selector_tuner_params
                    )
        elif self._train_flag == ConstantValues.increment:
            model_params.decay_rate = params[ConstantValues.decay_rate]
            model_params.init_model_root = None
            model_params.metric_eval_used_flag = False
        else:
            model_params.init_model_root = None
            model_params.increment_flag = params[ConstantValues.increment_flag]
            model_params.infer_result_type = params[ConstantValues.infer_result_type]
            model_params.metric_eval_used_flag = False
            self.__result = None

        self.__model = self.create_entity(entity_name=self.__model_name, **model_params)

    def _train_run(self, **entity):
        assert ConstantValues.train_dataset in entity.keys()
        assert ConstantValues.val_dataset in entity.keys()

        entity[ConstantValues.model] = self.__model
        entity[ConstantValues.metric] = self.__metric
        entity[ConstantValues.auto_ml] = self.__auto_ml
        entity[ConstantValues.feature_configure] = self.__feature_conf
        entity[ConstantValues.loss] = self.__loss

        if self.__feature_selector_flag is True:
            if self.__supervised_selector_mode == "model_select":
                self.__feature_selector.run(**entity)
            else:
                entity[ConstantValues.selector_model] = self.__selector_model
                entity[ConstantValues.selector_auto_ml] = self.__selector_auto_ml
                entity[ConstantValues.selector_metric] = self.__selector_metric
                self.__feature_selector.run(**entity)
        else:
            train_dataset = entity[ConstantValues.train_dataset]
            self.__metric.label_name = train_dataset.get_dataset().target_names[0]

            feature_conf = yaml_read(self.__feature_configure_path)
            self.__feature_conf.file_path = self.__feature_configure_path
            self.__feature_conf.parse(method="system")
            # if feature_list is None, all feature's used will be set true.
            self.__feature_conf.feature_select(feature_list=None)

            entity[ConstantValues.model].update_feature_conf(feature_conf=feature_conf)

            logger.info(
                "Auto machine learning component has started, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            self.__auto_ml.run(**entity)

            yaml_write(yaml_dict=feature_conf, yaml_file=self.__final_file_path)

        # self.__best_metric is a MetricResult object.
        self.__best_metric = entity[ConstantValues.model].val_best_metric_result

        logger.info(
            "Using %s, num of running auto machine learning train methods is : %s",
            self.__model_name, entity[ConstantValues.auto_ml].train_method_count
        )

        logger.info(
            "Using %s, num of running auto machine learning algorithms trials is : %s",
            self.__model_name, entity["auto_ml"].algorithm_method_count
        )

        logger.info(
            "Using %s, num of running auto machine learning model training trials is : %s",
            self.__model_name, entity["auto_ml"].trial_count
        )

        logger.info(
            "Using %s, all training model metric results are : %s",
            self.__model_name, entity["model"].metric_history
        )

        logger.info("Using {}, metric: {}, maximize metric result is : {:.10f}".format(
            self.__model_name, self.__metric_name, max(entity["model"].metric_history)
        )
        )

        logger.info("Using {}, metric: {}, minimize metric result is : {:.10f}".format(
            self.__model_name, self.__metric_name, min(entity["model"].metric_history)
        )
        )

        logger.info("Using {}, metric: {}, best metric result is : {:.10f}".format(
            self.__model_name, self.__metric_name, entity["model"].val_best_metric_result.result
        )
        )

        logger.info("Using {}, num of total models is {:d}".format(
            self.__model_name, len(entity["model"].metric_history)
        )
        )
        entity["model"].model_save()

    def _increment_run(self, **entity):
        assert "increment_dataset" in entity.keys()

        entity["model"] = self.__model
        entity["feature_configure"] = self.__feature_conf

        feature_conf = yaml_read(self.__feature_configure_path)
        self.__feature_conf.file_path = self.__feature_configure_path
        self.__feature_conf.parse(method="system")
        # if feature_list is None, all feature's used will be set true.
        self.__feature_conf.feature_select(feature_list=None)

        entity["model"].update_feature_conf(feature_conf=feature_conf)

        logger.info(
            "Incremental training has started, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        entity["model"].run(**entity)

        yaml_write(yaml_dict=feature_conf, yaml_file=self.__final_file_path)

        # self._best_metric is a MetricResult object.
        self._best_metric = entity["model"].val_best_metric_result

        entity["model"].model_save()

    def _predict_run(self, **entity):
        """
        :param entity:
        :return: This method will return predict result for test dataset.
        """
        assert "infer_dataset" in entity
        assert self._train_flag == ConstantValues.inference

        assert self.__final_file_path

        entity["feature_configure"] = self.__feature_conf
        entity["feature_configure"].file_path = self.__final_file_path
        entity["feature_configure"].parse(method="system")

        dataset = entity["infer_dataset"]
        feature_config = entity["feature_configure"]

        self.__model.update_feature_conf(feature_conf=feature_config)

        self.__result = self.__model.run(infer_dataset=dataset)

    @property
    def optimal_metric(self):
        """
        Best metric for this graph
        :return: MetricResult
        """
        assert self._train_flag
        assert self.__best_metric is not None
        return self.__best_metric

    @property
    def result(self):
        """
        :return: inference result, pd.Dataframe
        """
        assert self._train_flag == ConstantValues.inference
        assert self.__result is not None
        return self.__result

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
