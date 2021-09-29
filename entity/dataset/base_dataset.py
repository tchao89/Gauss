# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
from __future__ import annotations

import abc

from typing import List
from entity.entity import Entity


class BaseDataset(Entity):
    def __init__(self,
                 name: str,
                 data_path: str,
                 task_name: str,
                 target_name=None,
                 memory_only=True):

        super(BaseDataset, self).__init__(name=name)

        assert isinstance(target_name, List) or target_name is None

        self._data_path = data_path
        self._target_name = target_name
        self._memory_only = memory_only
        self._column_size = 0
        self._row_size = 0
        self._default_print_size = 5
        self._task_name = task_name

        # Bunch object, including features, target,
        # feature_names[optional], target_names[optional]
        self._bunch = None

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
    def column_size(self):
        return self._column_size

    @property
    def row_size(self):
        return self._row_size

    @property
    def target_name(self):
        return self._target_name

    @property
    def default_print_size(self):
        return self._default_print_size

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
    def feature_choose(self, feature_list):
        """eliminate features which are not selected."""
        pass
