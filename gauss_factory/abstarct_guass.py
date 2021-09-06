# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
"""
Abstract factory
"""
import abc

class AbstractGauss(metaclass=abc.ABCMeta):
    """
    AbstractGauss object
    """
    @abc.abstractmethod
    def get_entity(self, entity_name: str):
        """
        :param entity_name:
        :return:
        """

    @abc.abstractmethod
    def get_component(self, component_name: str):
        """
        :param component_name:
        :return:
        """
