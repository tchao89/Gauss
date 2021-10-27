"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import os
import json

from core.nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from core.nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from gauss.auto_ml.base_auto_ml import BaseAutoML
from entity.dataset.base_dataset import BaseDataset
from entity.model.model import ModelWrapper
from entity.metrics.base_metric import BaseMetric, MetricResult
from entity.losses.base_loss import BaseLoss

from utils.Logger import logger
from utils.base import get_current_memory_gb
from utils.constant_values import ConstantValues


class TabularAutoML(BaseAutoML):
    """
    TabularAutoML object
    """

    def __init__(self, **params):
        """
        :param name:
        :param train_flag:
        :param enable:
        :param opt_model_names: opt_model is a list object,
        and can includes tpe, random_search, anneal and evolution.
        """
        super().__init__(
            name=params[ConstantValues.name],
            train_flag=params[ConstantValues.train_flag],
            enable=params[ConstantValues.enable],
            task_name=params[ConstantValues.task_name],
            opt_model_names=params[ConstantValues.opt_model_names])

        assert params[ConstantValues.optimize_mode] in [ConstantValues.minimize,
                                                        ConstantValues.maximize]

        self.__opt_tuners = []
        # optional: "maximize", "minimize", depends on metrics for auto ml.
        self.__optimize_mode = params[ConstantValues.optimize_mode]
        # trial num for auto ml.
        self.__trial_num = params[ConstantValues.auto_ml_trial_num]
        self.__auto_ml_path = params[ConstantValues.auto_ml_path]
        self.__default_parameters = None
        self.__search_space = None

        self.__automl_final_set = True

        # self._model should be entity.model.Model object
        self.__model = None
        self.__best_metrics = None

        self.__local_best = None

        self.__chose_tuner_set()
        self.__set_search_space()
        self.__set_default_params()

    def __chose_tuner_set(self):
        """This method will fill self.opt_tuners,
        which contains all opt tuners you need in this experiment.
        :return: None
        """
        if not self._opt_model_names:
            raise ValueError("Object opt_model_names is empty.")

        if not self.__opt_tuners:
            for opt_name in self._opt_model_names:
                opt_tuner = self._chose_tuner(opt_name)
                self.__opt_tuners.append(opt_tuner)
        else:
            logger.info("Opt tuners have set successfully.")

    def _chose_tuner(self, algorithms_name: str):

        if algorithms_name == "tpe":
            return HyperoptTuner(algorithm_name="tpe", optimize_mode=self.__optimize_mode)
        if algorithms_name == "random_search":
            return HyperoptTuner(algorithm_name="random_search", optimize_mode=self.__optimize_mode)
        if algorithms_name == "anneal":
            return HyperoptTuner(algorithm_name="anneal", optimize_mode=self.__optimize_mode)
        if algorithms_name == "evolution":
            return EvolutionTuner(optimize_mode=self.__optimize_mode)

        raise RuntimeError('Not support tuner algorithm in tabular auto-ml algorithms.')

    @property
    def automl_final_set(self):
        """
        Decide whether executing model.set_best_model() here, if True,
        this method will execute here, otherwise in supervised feature selector component.
        :return: bool
        """
        return self.__automl_final_set

    @automl_final_set.setter
    def automl_final_set(self, final_set: bool):
        self.__automl_final_set = final_set

    def _train_run(self, **entity):
        self._train_method_count += 1

        assert "model" in entity and isinstance(entity["model"], ModelWrapper)
        assert "train_dataset" in entity and isinstance(entity["train_dataset"], BaseDataset)
        assert "val_dataset" in entity and isinstance(entity["val_dataset"], BaseDataset)
        assert "metric" in entity and isinstance(entity["metric"], BaseMetric)
        assert "loss" in entity
        assert isinstance(entity["loss"], BaseLoss) if entity["loss"] is not None else True

        self.__model = entity["model"]

        self.__local_best = None

        # 在此处创建模型数据对象, 继承entity对象, 放进entity字典
        for tuner in self.__opt_tuners:
            self._algorithm_method_count += 1

            logger.info(
                "Starting update search space, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            tuner.update_search_space(search_space=self.__search_space.get(entity["model"].name))
            for trial in range(self.__trial_num):
                self._trial_count += 1

                logger.info(
                    "tuner algorithms: {}, Auto machine learning trial number: {:d}".format(
                        tuner.algorithm_name, trial
                    )
                )
                if self.__default_parameters is not None:

                    params = self.__default_parameters.get(self.__model.name)
                    assert params is not None

                    receive_params = tuner.generate_parameters(trial)

                    logger.info(
                        "Generate hyper parameters, "
                        "with current memory usage: {:.2f} GiB".format(
                            get_current_memory_gb()["memory_usage"]
                        )
                    )
                    params.update(receive_params)

                    logger.info(
                        "Send parameters to model object, and update feature configure, "
                        "with current memory usage: {:.2f} GiB".format(
                            get_current_memory_gb()["memory_usage"]
                        )
                    )
                    self.__model.update_params(**params)

                    logger.info(
                        "Model training, with current memory usage: {:.2f} GiB".format(
                            get_current_memory_gb()["memory_usage"]
                        )
                    )

                    logger.info(
                        "Model training, with current memory usage: {} GiB".format(
                            params.keys()
                        )
                    )
                    self.__model.run(**entity)

                    logger.info(
                        "Update best model, with current memory usage: {:.2f} GiB".format(
                            get_current_memory_gb()["memory_usage"]
                        ))
                    self.__model.update_best_model()

                    metric = self.__model.val_metric

                    self.__update_local_best(metric)

                    metric_result = self.__model.val_metric.result
                    tuner.receive_trial_result(trial, receive_params, metric_result)
                else:
                    raise ValueError("Default parameters is None.")

        if self.__automl_final_set is True:
            self.__model.set_best_model()
        self.__best_metric = self.__model.val_best_metric_result.result

    def _increment_run(self, **entity):
        self._train_run(**entity)

    def _predict_run(self, **entity):
        pass

    def __update_local_best(self, metric):
        """
        This method will update model, model parameters and
        validation metric result in each training.
        :return: None
        """
        if self.__local_best is None:
            self.__local_best = MetricResult(
                name=metric.name,
                metric_name=metric.metric_name,
                result=metric.result,
                optimize_mode=metric.optimize_mode
            )

        if self.__local_best.__cmp__(metric) < 0:
            self.__local_best = MetricResult(
                name=metric.name,
                metric_name=metric.metric_name,
                result=metric.result,
                optimize_mode=metric.optimize_mode
            )

    @property
    def local_best(self):
        """
        Get best metric result of single trial in auto machine learning.
        :return: MetricResult
        """
        return self.__local_best

    @property
    def optimal_metric(self):
        """
        :return: MetricResult.result
        """
        return self.__best_metric

    @property
    def default_params(self):
        """
        Get default parameters.
        :return: dict
        """
        return self.__default_parameters

    def __set_default_params(self):
        """
        Load default parameters.
        :return: None
        """
        default_params_path = os.path.join(self.__auto_ml_path, "default_parameters.json")
        with open(default_params_path, 'r', encoding="utf-8") as json_file:
            self.__default_parameters = json.load(json_file)

    @property
    def search_space(self):
        """
        Get search space configuration.
        :return: dict
        """
        return self.__search_space

    def __set_search_space(self):
        """
        Load search space configuration.
        :return: None
        """
        search_space_path = os.path.join(self.__auto_ml_path, "search_space.json")
        with open(search_space_path, 'r', encoding="utf-8") as json_file:
            self.__search_space = json.load(json_file)

    @property
    def model(self):
        """

        :return: Model object
        """
        return self.__model
