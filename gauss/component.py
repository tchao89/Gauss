"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: luoqing
component object, base class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc

from utils.constant_values import ConstantValues


class Component(metaclass=abc.ABCMeta):
    """Base class for a component of ML workflow, such as feature generation, feature selector
    and so on.
        All subclasses of Component must override the _train_run() method and
    _inference_run() method.
    """

    def __init__(self,
                 name: str,
                 train_flag: bool = True,
                 enable: bool = True,
                 task_name: str = None):
        """Construct a Component.

        :param name: The name of the Component.
        :param train_flag: The flag of train or inference statues of current workflow
        """
        assert task_name in [ConstantValues.binary_classification,
                             ConstantValues.multiclass_classification,
                             ConstantValues.regression]

        self._task_name = task_name
        self._name = name
        self._train_flag = train_flag
        self._enable = enable

    @property
    def name(self):
        """
        Get name.
        :return: String
        """
        return self._name

    @property
    def enable(self):
        """
        Get enable
        :return: bool
        """
        return self._enable

    @property
    def train_flag(self):
        """
        Get train flag.
        :return: bool
        """
        return self._train_flag

    def task_name(self):
        """
        Get task name.
        :return: string
        """
        return self._task_name

    def run(self, **entity):
        """
        Run component.
        :param entity:
        :return:
        """
        if self._train_flag == ConstantValues.train:
            self._train_run(**entity)
        if self._train_flag == ConstantValues.inference:
            self._predict_run(**entity)
        if self._train_flag == ConstantValues.increment:
            self._increment_run(**entity)

    @abc.abstractmethod
    def _train_run(self, **entity):
        pass

    @abc.abstractmethod
    def _predict_run(self, **entity):
        pass

    @abc.abstractmethod
    def _increment_run(self, **entity):
        pass
