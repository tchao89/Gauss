# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gauss.model.model import Model


class Gbdt(Model):
    """gbdt model, include xgboost, lightgbm, catboost
    """

    def __init__(self, name: str, metric_name: str):
        super().__init__(name, metric_name)

    def model_save(self):
        pass
