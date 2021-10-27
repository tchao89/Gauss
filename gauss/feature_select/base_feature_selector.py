"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import abc

from gauss.component import Component


class BaseFeatureSelector(Component):
    """
    BaseFeatureSelector object.
    """
    def __init__(self,
                 name: str,
                 train_flag: str,
                 enable: bool,
                 task_name: str,
                 feature_configure_path):

        super(BaseFeatureSelector, self).__init__(name=name,
                                                  train_flag=train_flag,
                                                  enable=enable,
                                                  task_name=task_name)
        self._feature_configure_path = feature_configure_path

    @abc.abstractmethod
    def _train_run(self, **entity):
        pass

    @abc.abstractmethod
    def _predict_run(self, **entity):
        pass

    @abc.abstractmethod
    def _increment_run(self, **entity):
        pass
