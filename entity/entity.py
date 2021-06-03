# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: luoqing

import abc

class Entity(metaclass=abc.ABCMeta):
    """Base class for a entity of ML workflow,include
       data for model, feature configure for data,and 
       model for training, so on ...
    """
    def __init__(self,
                 name: str):
        """Construct a Entity.

        :param name: The name of the Entity.
        """
        self._name = name

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def __repr__(self):
        pass
