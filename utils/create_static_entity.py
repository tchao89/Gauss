"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
from gauss_factory.gauss_factory_producer import GaussFactoryProducer


def create_static_entity(entity_name: str):

    gauss_factory = GaussFactoryProducer()
    entity_factory = gauss_factory.get_factory(choice="static_entity")
    return entity_factory.get_entity(entity_name=entity_name)
