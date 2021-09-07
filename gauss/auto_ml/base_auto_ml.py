"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import abc
from gauss.component import Component


class BaseAutoML(Component):
    def __init__(self,
                 name: str,
                 train_flag: bool,
                 enable: bool,
                 task_name: str,
                 opt_model_names: list = None
                 ):
        super().__init__(
            name=name,
            train_flag=train_flag,
            enable=enable,
            task_name=task_name
        )

        self._opt_model_names = opt_model_names

    @abc.abstractmethod
    def _train_run(self, **entity):
        pass

    @abc.abstractmethod
    def _predict_run(self, **entity):
        pass
