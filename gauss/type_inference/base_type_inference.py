# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
import abc

from gauss.component import Component
from entity.dataset.base_dataset import BaseDataset


class BaseTypeInference(Component):
    def __init__(self,
                 name: str,
                 train_flag: bool,
                 task_name: str,
                 source_file_path="null",
                 final_file_path: str = './',
                 final_file_prefix="final"):

        self._source_file_path = source_file_path
        self._final_file_path = final_file_path
        self._final_file_prefix = final_file_prefix
        self._update_flag = False
        super(BaseTypeInference, self).__init__(
            name=name,
            train_flag=train_flag,
            task_name=task_name
        )

    @abc.abstractmethod
    def dtype_inference(self, dataset: BaseDataset):
        pass

    @abc.abstractmethod
    def ftype_inference(self, dataset: BaseDataset):
        pass

    @abc.abstractmethod
    def target_check(self, target: BaseDataset):
        pass

    @property
    def source_file_path(self):
        return self._source_file_path

    @property
    def target_file_path(self):
        return self._final_file_path

    @property
    def target_file_prefix(self):
        return self._final_file_prefix

    def _train_run(self, **entity):
        pass

    def _predict_run(self, **entity):
        pass
