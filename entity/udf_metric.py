# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic Inc. All rights reserved.
# Authors: Lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import roc_auc_score

from entity.base_metric import BaseMetric
from entity.base_metric import MetricResult


class AUC(BaseMetric):

    def __init__(self, **params):
        super().__init__(name=params["name"])
        self._label_name = params["label_name"]
        self._metrics_result = None

    def __repr__(self):
        print("go")

    def set_label_name(self, label_name: str):
        self._label_name = label_name

    def set_name(self, name: str):
        self._name = name

    def evaluate(self, predict, labels_map):
        if len(labels_map.shape) > 1:
            label = labels_map.loc[:, [self._label_name]]
        else:
            label = labels_map

        if np.sum(label) == 0 or np.sum(label) == label.size:
            self._metrics_result = MetricResult(name=self.name, result=float('nan'))
        else:
            auc = roc_auc_score(y_true=label, y_score=predict)
            self._metrics_result = MetricResult(name=self.name, result=auc, meta={'#': predict.size})

        return self._metrics_result

    @property
    def required_label_names(self):
        return [self._label_name]

    @property
    def metrics_result(self):
        assert self._metrics_result is not None
        return self._metrics_result
