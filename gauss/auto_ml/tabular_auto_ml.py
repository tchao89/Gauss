# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab

from core.nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from core.nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from gauss.auto_ml.base_auto_ml import BaseAutoML
from entity.base_dataset import BaseDataset
from entity.model import Model
from entity.base_metric import BaseMetric

class TabularAutoML(BaseAutoML):
    def __init__(self, name: str, train_flag: bool = True, enable: bool = True, opt_model_names=None, default_params=None):
        """

        :param name:
        :param train_flag:
        :param enable:
        :param opt_model_names: opt_model is a list object, and can includes tpe, random_search, anneal and evolution.
        """
        super(TabularAutoML, self).__init__(name, train_flag, enable, opt_model_names)
        self.opt_tuners = []
        # optional: "maximize", "minimize", depends on metrics for auto ml.
        self.optimize_mode = "maximize"
        # trial num for auto ml.
        self.trial_num = 15
        self.default_params = default_params

    def chose_tuner_set(self):
        """This method will fill self.opt_tuners, which contains all opt tuners you need in this experiment.

        :return: None
        """
        if not self.opt_model_names:
            raise ValueError("Object opt_model_names is empty.")
        else:
            for opt_name in self.opt_model_names:
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

    @property
    def default_params(self):
        return self.default_params

    @default_params.setter
    def default_params(self, params: dict):
        self.default_params = params

    def _train_run(self, **entity):
        assert "model" in entity and isinstance(entity["model"], Model)
        assert "dataset" in entity and isinstance(entity["dataset"], BaseDataset)
        assert "metrics" in entity and isinstance(entity["metrics"], BaseMetric)
        assert "search_space" in entity

        self.chose_tuner_set()

        for tuner_algorithms in self.opt_tuners:
            tuner = tuner_algorithms
            tuner.update_search_space(entity["search_space"])

            for trial in range(self.trial_num):
                params = self.get_default_params()
                receive_params = tuner.generate_parameters(trial)
                params.update(receive_params)
                # 更新模型参数
                model = entity["model"]
                model.update_params(params)
                model.train(**entity)
                metrics = model.get_train_metric()
                tuner.receive_trial_result(trial, receive_params, metrics)

    def _predict_run(self, **entity):
        pass
