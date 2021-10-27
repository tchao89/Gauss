# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
from gauss.component import Component


class BaseFeatureGenerator(Component):
    def __init__(self, name: str, train_flag: str, enable: bool, task_name: str, feature_configure_path):
        super(BaseFeatureGenerator, self).__init__(name=name,
                                                   train_flag=train_flag,
                                                   enable=enable,
                                                   task_name=task_name)
        self._feature_configure_path = feature_configure_path

    def _train_run(self, **entity):
        pass

    def _predict_run(self, **entity):
        pass

    def _increment_run(self, **entity):
        pass
