# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
from gauss.component import Component


class BaseTypeInference(Component):
    def __init__(self,
                 name: str,
                 train_flag: bool,
                 source_file_path="null",
                 target_file_path: str = './',
                 target_file_prefix="target"):

        self._source_file_path = source_file_path
        self._target_file_path = target_file_path
        self._target_file_prefix = target_file_prefix
        self._update_flag = False
        super(BaseTypeInference, self).__init__(
            name=name,
            train_flag=train_flag
        )

    @property
    def source_file_path(self):
        return self._source_file_path

    @property
    def target_file_path(self):
        return self._target_file_path

    @property
    def target_file_prefix(self):
        return self._target_file_prefix

    def _train_run(self, **entity):
        pass

    def _predict_run(self, **entity):
        pass
