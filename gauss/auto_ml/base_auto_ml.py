# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import abc
from gauss.component import Component


class BaseAutoML(Component):
    def __init__(self, name: str, train_flag: bool = True, enable: bool = True, opt_model_names: list = None):
        super(BaseAutoML, self).__init__(name, train_flag, enable)
        self.opt_model_names = opt_model_names

    @abc.abstractmethod
    def _train_run(self, **entity):
        pass

    @abc.abstractmethod
    def _predict_run(self, **entity):
        pass
