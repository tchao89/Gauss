# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
from gauss_factory.gauss_factory_producer import GaussFactoryProducer


def check_data(already_data_clear, model_name):
    assert isinstance(already_data_clear, bool)
    assert isinstance(model_name, str)

    if already_data_clear is not True:
        static_entity = create_static_entity(model_name)
        assert static_entity is not None
        if static_entity.need_data_clear is True:
            return False
    return True
