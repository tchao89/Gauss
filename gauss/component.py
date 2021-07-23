# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: luoqing

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc

class Component(metaclass=abc.ABCMeta):
    """Base class for a component of ML workflow, such as feature  generation,
    feature selector and so on     
    All subclasses of Component must override the _train_run() method
    and _inference_run() method
    """

    def __init__(self,
                 name: str,
                 train_flag: bool = True,
                 enable: bool = True):
        """Construct a Component.

        :param name: The name of the Component.
        :param train_flag: The flag of train or inference statues of 
               current workflow
        """
        self._name = name
        self._train_flag = train_flag
        self._enable = enable

    @property
    def name(self):
        return self._name

    @property
    def enable(self):
        return self._enable

    @property
    def train_flag(self):
        return self._train_flag
    
    def run(self, **entity):
        if self._train_flag:
            self._train_run(**entity)
        else:
            self._predict_run(**entity)

    @abc.abstractmethod
    def _train_run(self, **entity):
        pass

    @abc.abstractmethod
    def _predict_run(self, **entity):
        pass
