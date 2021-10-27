"""-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab"""
from __future__ import annotations

import abc

from entity.entity import Entity


class BaseDataset(Entity):
    def __init__(self,
                 name: str,
                 data_path: str,
                 task_name: str,
                 train_flag: str,
                 memory_only=True):

        super(BaseDataset, self).__init__(name=name)

        self._data_path = data_path
        self._memory_only = memory_only
        self._task_name = task_name
        self._train_flag = train_flag

    @abc.abstractmethod
    def load_data(self):
        """load data from file provided."""
        pass

    @abc.abstractmethod
    def get_dataset(self):
        """return loaded data."""
        pass

    @abc.abstractmethod
    def set_dataset(self, data_pair):
        """change dataset"""

    @property
    def task_type(self):
        return self._task_name

    @abc.abstractmethod
    def split(self):
        pass

    @abc.abstractmethod
    def union(self, dataset: BaseDataset):
        pass

    @abc.abstractmethod
    def feature_choose(self, feature_list, use_index_flag):
        """eliminate features which are not selected."""
        pass
