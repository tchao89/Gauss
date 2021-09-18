# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic Inc. All rights reserved.
# Authors: Lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import roc_auc_score

from entity.metrics.base_metric import BaseMetric
from entity.metrics.base_metric import MetricResult


class AUC(BaseMetric):

    def __init__(self, **params):
        super().__init__(name=params["name"], optimize_mode="maximize")
        self._label_name = params.get("label_name")
        self._metrics_result = None

    def __repr__(self):
        if self._metrics_result is None:
            return "{name} metric with {optimize_mode} optimize mode."\
                .format(
                    name=self._name,
                    optimize_mode=self._optimize_mode
                )
        else:
            return "{name} metric with {optimize_mode} optimize mode, result is: {result}."\
                .format(
                    name=self._name,
                    optimize_mode=self._optimize_mode,
                    result=self._metrics_result
                ) 

    @property
    def label_name(self):
        return self._label_name

    @label_name.setter
    def label_name(self, label_name: str):
        self._label_name = label_name

    def evaluate(self, predict: np.ndarray, labels_map: np.ndarray):
        """
        :param predict: np.ndarray object, (n_sample)
        :param labels_map: np.ndarray object, (n_samples)
        :return: MetricResult object
        """
        if np.sum(labels_map) == 0 or np.sum(labels_map) == labels_map.shape[0]:
            self._metrics_result = MetricResult(name=self.name, result=float('nan'), optimize_mode=self._optimize_mode)
        else:
            auc = roc_auc_score(y_true=labels_map, y_score=predict)
            self._metrics_result = MetricResult(name=self.name, result=auc, meta={'#': predict.size},
                                                optimize_mode=self._optimize_mode)

        return self._metrics_result

    @property
    def required_label_names(self):
        return [self._label_name]

    @property
    def metrics_result(self):
        assert self._metrics_result is not None
        return self._metrics_result


class NNAUC(BaseMetric):

    def __init__(self, **params):
        super().__init__(
            name=params["name"], 
            optimize_mode="maximize"
            )
        self._label_name = params.get("label_name")
        self._metrics_result = None

    def __repr__(self):
        if self._metrics_result is None:
            return "{name} metric with {optimize_mode} optimize mode."\
                .format(
                    name=self._name,
                    optimize_mode=self._optimize_mode
                )
        else:
            return "{name} metric with {optimize_mode} optimize mode, result is: {result}."\
                .format(
                    name=self._name,
                    optimize_mode=self._optimize_mode,
                    result=self._metrics_result
                ) 
        

    def evaluate(self, predict, labels_map):
        """
        :param predict: np.ndarray object, (n_sample)
        :param labels_map: Dict[List], true value.

        :return: MetricResult object
        """
        label = labels_map[self._label_name[0]]
        if np.sum(label) == 0 or np.sum(label) == label.shape[0]:
            self._metrics_result = MetricResult(name=self.name, result=float('nan'))
        else:
            auc = roc_auc_score(y_true=label, y_score=predict)
            self._metrics_result = MetricResult(
                name=self.name, 
                result=auc, 
                optimize_mode=self._optimize_mode, 
                meta={'#': predict.size}
                )
        return self._metrics_result
    
    
    @property
    def label_name(self):
        return self._label_name

    @label_name.setter
    def label_name(self, label_name: str):
        self._label_name = [label_name]

    @property
    def required_label_names(self):
        return [self._label_name]

    @property
    def metrics_result(self):
        assert self._metrics_result is not None
        return self._metrics_result

    def metric_result(self):
        pass