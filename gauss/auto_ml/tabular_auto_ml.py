# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import os
import json

from core.nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from core.nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from gauss.auto_ml.base_auto_ml import BaseAutoML
from entity.base_dataset import BaseDataset
from entity.model import Model
from entity.base_metric import BaseMetric


class TabularAutoML(BaseAutoML):
    def __init__(self, **params):
        """
        :param name:
        :param train_flag:
        :param enable:
        :param opt_model_names: opt_model is a list object, and can includes tpe, random_search, anneal and evolution.
        """
        super(TabularAutoML, self).__init__(params["name"], params["train_flag"], params["enable"], params["opt_model_names"])
        self.opt_tuners = []
        # optional: "maximize", "minimize", depends on metrics for auto ml.
        self.optimize_mode = "maximize"
        # trial num for auto ml.
        self.trial_num = 5
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

    @classmethod
    def _chose_tuner(cls, algorithms_name: str):

        if algorithms_name == "tpe":
            return HyperoptTuner("tpe")
        if algorithms_name == "random_search":
            return HyperoptTuner("random_search")
        if algorithms_name == "anneal":
            return HyperoptTuner("anneal")
        if algorithms_name == "evolution":
            return EvolutionTuner()
        else:
            raise RuntimeError('Not support tuner algorithm in tabular auto-ml algorithms.')

    def _train_run(self, **entity):

        assert "model" in entity and isinstance(entity["model"], Model)
        assert "dataset" in entity and isinstance(entity["dataset"], BaseDataset)
        assert "val_dataset" in entity and isinstance(entity["dataset"], BaseDataset)
        assert "metrics" in entity and isinstance(entity["metrics"], BaseMetric)

        self.chose_tuner_set()
        self.set_search_space()
        self.set_default_params()

        for tuner_algorithms in self.opt_tuners:
            tuner = tuner_algorithms
            tuner.update_search_space(self._search_space)

            for trial in range(self.trial_num):
                if self._default_parameters is not None:
                    params = self._default_parameters
                    receive_params = tuner.generate_parameters(trial)
                    params.update(receive_params)
                    model = entity["model"]
                    model.update_params(**params)
                    model.train(**entity)
                    metrics = model.val_metrics.result
                    tuner.receive_trial_result(trial, receive_params, metrics)
                else:
                    raise ValueError("Default parameters is None.")

    def _predict_run(self, **entity):
        pass

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
        print("self._auto_ml_path", self._auto_ml_path)
        search_space_path = os.path.join(self._auto_ml_path, "search_space.json")
        print(search_space_path)
        with open(search_space_path, 'r') as json_file:
            self._search_space = json.load(json_file)
