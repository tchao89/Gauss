"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
from utils.create_static_entity import create_static_entity


def check_data(already_data_clear, model_need_clear_flag):
    assert isinstance(already_data_clear, bool)
    assert isinstance(model_need_clear_flag,  bool)

    if already_data_clear is False:
        if model_need_clear_flag is True:
            return False
    return True
