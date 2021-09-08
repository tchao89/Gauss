# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import os
import json

from core.nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from core.nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from gauss.auto_ml.base_auto_ml import BaseAutoML
from entity.dataset.base_dataset import BaseDataset
from entity.model.model import ModelWrapper
from entity.metrics.base_metric import BaseMetric, MetricResult

from utils.Logger import logger
from utils.base import get_current_memory_gb


class TabularAutoML(BaseAutoML):
    def __init__(self, **params):
        """
        :param name:
        :param train_flag:
        :param enable:
        :param opt_model_names: opt_model is a list object, and can includes tpe, random_search, anneal and evolution.
        """
        super().__init__(
            name=params["name"],
            train_flag=params["train_flag"],
            enable=params["enable"],
            task_name=params["task_name"],
            opt_model_names=params["opt_model_names"])

        assert "optimize_mode" in params
        assert params["optimize_mode"] in ["minimize", "maximize"]

        self.opt_tuners = []
        # optional: "maximize", "minimize", depends on metrics for auto ml.
        self._optimize_mode = params["optimize_mode"]
        # trial num for auto ml.
        self.trial_num = params["auto_ml_trial_num"]
        self._auto_ml_path = params["auto_ml_path"]
        self._default_parameters = None
        self._search_space = None

        self._is_final_set = True
        # self._model should be entity.model.Model object
        self._model = None
        self._best_metrics = None

        self._result = None
        self._multi_process_result = []

        self._local_best = None

    def chose_tuner_set(self):
        """This method will fill self.opt_tuners, which contains all opt tuners you need in this experiment.

        :return: None
        """
        if not self._opt_model_names:
            raise ValueError("Object opt_model_names is empty.")

        if not self.opt_tuners:
            for opt_name in self._opt_model_names:
                opt_tuner = self._chose_tuner(opt_name)
                self.opt_tuners.append(opt_tuner)
        else:
            logger.info("Opt tuners have set.")

    def _chose_tuner(self, algorithms_name: str):

        if algorithms_name == "tpe":
            return HyperoptTuner(algorithm_name="tpe", optimize_mode=self._optimize_mode)
        if algorithms_name == "random_search":
            return HyperoptTuner(algorithm_name="random_search", optimize_mode=self._optimize_mode)
        if algorithms_name == "anneal":
            return HyperoptTuner(algorithm_name="anneal", optimize_mode=self._optimize_mode)
        if algorithms_name == "evolution":
            return EvolutionTuner(optimize_mode=self._optimize_mode)
        else:
            raise RuntimeError('Not support tuner algorithm in tabular auto-ml algorithms.')

    @property
    def is_final_set(self):
        return self._is_final_set

    @is_final_set.setter
    def is_final_set(self, final_set: bool):
        self._is_final_set = final_set

    def _train_run(self, **entity):
        self._train_method_count += 1

        assert "model" in entity and isinstance(entity["model"], ModelWrapper)
        assert "train_dataset" in entity and isinstance(entity["train_dataset"], BaseDataset)
        assert "val_dataset" in entity and isinstance(entity["val_dataset"], BaseDataset)
        assert "metrics" in entity and isinstance(entity["metrics"], BaseMetric)

        self.chose_tuner_set()
        self.set_search_space()
        self.set_default_params()

        self._model = entity["model"]

        self._local_best = None

        # 在此处创建模型数据对象, 继承entity对象, 放进entity字典
        for tuner in self.opt_tuners:
            self._algorithm_method_count += 1

            assert len(self.opt_tuners) < 5, "Length of opt tuners is {:d}.".format(len(self.opt_tuners))
            logger.info(
                "Starting update search space, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            tuner.update_search_space(self._search_space.get(entity["model"].name))
            for trial in range(self.trial_num):
                self._trial_count += 1

                logger.info(
                    "tuner algorithms: " + tuner.algorithm_name + ", Auto machine learning trial number: %d",
                    trial)
                if self._default_parameters is not None:

                    params = self._default_parameters.get(self._model.name)
                    assert params is not None

                    receive_params = tuner.generate_parameters(trial)

                    logger.info("Generate hyper parameters, with current memory usage: %.2f GiB",
                                get_current_memory_gb()["memory_usage"])
                    params.update(receive_params)

                    logger.info("Send parameters to model object, with current memory usage: %.2f GiB",
                                get_current_memory_gb()["memory_usage"])
                    self._model.update_params(**params)

                    logger.info("Model training, with current memory usage: %.2f GiB",
                                get_current_memory_gb()["memory_usage"])
                    self._model.train(**entity)

                    logger.info("Evaluate model, with current memory usage: %.2f GiB",
                                get_current_memory_gb()["memory_usage"])
                    self._model.eval(**entity)

                    logger.info("Update best model, with current memory usage: %.2f GiB",
                                get_current_memory_gb()["memory_usage"])
                    self._model.update_best_model()

                    metrics = self._model.val_metrics

                    self.__update_local_best(metrics)

                    metrics_result = self._model.val_metrics.result
                    tuner.receive_trial_result(trial, receive_params, metrics_result)
                else:
                    raise ValueError("Default parameters is None.")

        if self._is_final_set is True:
            self._model.set_best_model()
        self._best_metrics = self._model.val_best_metric_result.result

    def _predict_run(self, **entity):
        pass

    def __update_local_best(self, metrics):
        """
        This method will update model, model parameters and
        validation metric result in each training.
        :return: None
        """
        if self.local_best is None:
            self._local_best = MetricResult(
                name="local_best",
                result=metrics.result,
                optimize_mode=metrics.optimize_mode
            )

        if self.local_best.__cmp__(metrics) < 0:
            self._local_best = MetricResult(
                name=metrics.name,
                result=metrics.result,
                optimize_mode=metrics.optimize_mode
            )

    @property
    def local_best(self):
        return self._local_best

    @property
    def optimal_metrics(self):
        """
        :return: MetricResult.result
        """
        return self._best_metrics

    @property
    def default_params(self):
        return self._default_parameters

    def set_default_params(self):
        default_params_path = os.path.join(self._auto_ml_path, "default_parameters.json")
        with open(default_params_path, 'r') as json_file:
            self._default_parameters = json.load(json_file)

    @property
    def search_space(self):
        return self.search_space()

    def set_search_space(self):
        search_space_path = os.path.join(self._auto_ml_path, "search_space.json")
        with open(search_space_path, 'r') as json_file:
            self._search_space = json.load(json_file)

    @property
    def model(self):
        return self._model
