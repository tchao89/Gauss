# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
"""
This object is used to produce gauss entity or component factory.
"""
from gauss_factory.entity_factory import EntityFactory
from gauss_factory.component_factory import ComponentFactory
from gauss_factory.entity_factory import StaticModelFactory


class GaussFactoryProducer:
    """
    GaussFactoryProducer object
    """
    @staticmethod
    def get_factory(choice: str):
        """
        Produce factory.
        :param choice:
        :return:
        """
        if choice.lower() == "entity":
            return EntityFactory()
        if choice.lower() == "component":
            return ComponentFactory()
        if choice.lower() == "static_entity":
            return StaticModelFactory()
        return None
