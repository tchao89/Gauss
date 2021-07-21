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
from entity.model.model import Model
from entity.metrics.base_metric import BaseMetric


class TabularAutoML(BaseAutoML):
    def __init__(self, **params):
        """
        :param name:
        :param train_flag:
        :param enable:
        :param opt_model_names: opt_model is a list object, and can includes tpe, random_search, anneal and evolution.
        """
        super(TabularAutoML, self).__init__(params["name"], params["train_flag"], params["enable"], params["opt_model_names"])

        assert "optimize_mode" in params
        assert params["optimize_mode"] in ["minimize", "maximize"]

        self.opt_tuners = []
        # optional: "maximize", "minimize", depends on metrics for auto ml.
        self._optimize_mode = params["optimize_mode"]
        # trial num for auto ml.
        self.trial_num = 15
        self._auto_ml_path = params["auto_ml_path"]
        self._default_parameters = None
        self._search_space = None

    def chose_tuner_set(self):
        """This method will fill self.opt_tuners, which contains all opt tuners you need in this experiment.

        :return: None
        """
        if not self._opt_model_names:
            raise ValueError("Object opt_model_names is empty.")

        else:
            for opt_name in self._opt_model_names:
                opt_tuner = self._chose_tuner(opt_name)
                self.opt_tuners.append(opt_tuner)

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

    def run(self, **entity):
        if self._train_flag:
            return self._train_run(**entity)
        else:
            return self._predict_run(**entity)

    def _train_run(self, **entity):

        assert "model" in entity and isinstance(entity["model"], Model)
        assert "dataset" in entity and isinstance(entity["dataset"], BaseDataset)
        assert "val_dataset" in entity and isinstance(entity["val_dataset"], BaseDataset)
        assert "metrics" in entity and isinstance(entity["metrics"], BaseMetric)

        self.chose_tuner_set()
        self.set_search_space()
        self.set_default_params()

        best_model = None
        best_metrics = None

        for tuner_algorithms in self.opt_tuners:
            tuner = tuner_algorithms
            tuner.update_search_space(self._search_space)

            for trial in range(self.trial_num):
                if self._default_parameters is not None:
                    model = entity["model"]
                    params = self._default_parameters.get(model.name)
                    assert params is not None

                    receive_params = tuner.generate_parameters(trial)
                    params.update(receive_params)
                    model.update_params(**params)
                    model.train(**entity)

                    model.eval(**entity)
                    metrics = model.val_metrics.result
                    # get model which has been trained.

                    if best_model is None:
                        best_model = model

                    if best_metrics is None:
                        best_metrics = metrics

                    if metrics > best_metrics:
                        best_model = model
                        best_metrics = metrics

                    tuner.receive_trial_result(trial, receive_params, metrics)
                else:
                    raise ValueError("Default parameters is None.")
        return best_model

    def _predict_run(self, **entity):
        pass

    @property
    def default_params(self):
        return self._default_parameters

    def set_default_params(self):
        default_params_path = os.path.join(self._auto_ml_path, "default_parameters.json")
        with open(default_params_path, 'r') as json_file:
            print(default_params_path)
            self._default_parameters = json.load(json_file)

    @property
    def search_space(self):
        return self.search_space()

    def set_search_space(self):
        search_space_path = os.path.join(self._auto_ml_path, "search_space.json")
        with open(search_space_path, 'r') as json_file:
            self._search_space = json.load(json_file)
