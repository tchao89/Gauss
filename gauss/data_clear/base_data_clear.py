# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import abc

from gauss.component import Component


class BaseDataClear(Component):
    def __init__(self, name, train_flag, enable, task_name):
        """
        如果输入数据中存在缺失值，其需要经过dataclear模块
        :param name:
        :param train_flag:
        :param enable:
        """
        super(BaseDataClear, self).__init__(name=name, train_flag=train_flag, enable=enable, task_name=task_name)

    @abc.abstractmethod
    def _train_run(self, **entity):
        pass

    @abc.abstractmethod
    def _predict_run(self, **entity):
        pass
