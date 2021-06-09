# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
from gauss.component import Component


class BaseDataClear(Component):
    def __init__(self, name, train_flag, enable):
        super(BaseDataClear, self).__init__(name=name, train_flag=train_flag, enable=enable)

    def _train_run(self, **entity):
        pass

    def _predict_run(self, **entity):
        pass
