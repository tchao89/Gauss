# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: luoqing

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
from typing import Callable, List


class Component(metaclass=abc.ABCMeta):
    """Base class for a component of ML workflow.
    All subclasses of Component must override the _train_run() method
    and _inference_run() method
    """

    def __init__(self,
                 name: str,
                 train_flag: bool = True):
        """Construct a Component.

        :param name: The name of the Component.
        :param train_flag: The flag of train or inference statues of 
               current workflow
        """
        self._name = name
        self._train_flag = train_flag

    def init(self, sess):
        pass

    @property
    def name(self):
        return self._name

    @property
    def train_flag(self):
        return self._train_flag

    
    def run(self):
        if self.train_flag:
            self._train_run()
        else:
            self._inference_run()

    @abc.abstractmethod
    def _train_run(self):
        pass

    @abc.abstractmethod
    def _inference_run(self):
        pass
  
