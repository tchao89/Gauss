"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""

def check_data(already_data_clear, model_need_clear_flag):
    assert isinstance(already_data_clear, bool), \
        "Value: already_data_clear must be bool type, " \
        "but get {}".format(type(already_data_clear))

    assert isinstance(model_need_clear_flag,  bool), \
        "An effective value: model_need_clear_flag can not found " \
        "in system by model name supplied by used"

    if already_data_clear is False:
        if model_need_clear_flag is True:
            return False
    return True
