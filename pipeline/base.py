# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab

def check_data(already_data_clear, model_name):
    assert isinstance(already_data_clear, bool)
    assert isinstance(model_name, str)

    if already_data_clear is not True:
        if model_name not in ["lightgbm", "xgboost", "catboost"]:
            return False
    return True
