# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
import abc

class AbstractGauss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_entity(self, entity_name: str):
        pass

    @abc.abstractmethod
    def get_component(self, component_name: str):
        pass
