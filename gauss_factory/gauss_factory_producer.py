# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
from gauss_factory.entity_factory import EntityFactory
from gauss_factory.component_factory import ComponentFactory
from gauss_factory.entity_factory import StaticModelFactory


class GaussFactoryProducer:
    @staticmethod
    def get_factory(choice: str):
        if choice.lower() == "entity":
            return EntityFactory()
        elif choice.lower() == "component":
            return ComponentFactory()
        elif choice.lower() == "static_entity":
            return StaticModelFactory()
        return None
