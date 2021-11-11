# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from abc import ABC
from typing import List

import numpy as np

from entity.entity import Entity


class MetricResult(Entity):
    """Class for metric result."""

    def __init__(self,
                 name: str,
                 metric_name: str,
                 result: float,
                 optimize_mode: str,
                 meta=None):
        """Construct a metric result.
        :param result: The metric's result.
        :param meta: A map of other meta metarmation.
        """
        assert optimize_mode is not None

        self._metric_name = metric_name
        self._optimize_mode = optimize_mode
        if meta is None:
            meta = {}
        self._result = result
        self._meta = meta

        super(MetricResult, self).__init__(
            name=name,
        )

    @property
    def metric_name(self):
        return self._metric_name

    @property
    def optimize_mode(self):
        return self._optimize_mode

    @property
    def result(self):
        return self._result

    @property
    def meta(self):
        return self._meta

    def __repr__(self):
        if self._meta:
            meta_str = ','.join(['%s:%s' % (k, v) for k, v in self._meta.items()])
            return "%f (%s)" % (self._result, meta_str)
        else:
            return "%f" % self._result

    def __cmp__(self, other):
        assert self._optimize_mode in ["maximize", "minimize"]

        if self._optimize_mode == "maximize":
            return self._result - other.result
        else:
            return - self._result + other.result


class BaseMetric(Entity, ABC):
    """Base class for a evaluative metric.

    All subclasses of BaseMetric must override `eval` and `required_label_names` methods
    to conduct the metric evaluation and provide names of required data fields.
    """

    def __init__(self, name: str, optimize_mode: str, label_name: str = None, meta=None):
        """Construct a metric result.
        :param meta: A map of other meta metarmation.
        """

        if meta is None:
            meta = {}

        assert optimize_mode in ["minimize", "maximize"]
        self._optimize_mode = optimize_mode

        self._meta = meta
        self._label_name = label_name
        super(BaseMetric, self).__init__(
            name=name,
        )

    @property
    def optimize_mode(self):
        return self._optimize_mode

    @abc.abstractmethod
    def evaluate(self,
                 predict: np.ndarray,
                 labels_map: dict) -> MetricResult:
        """Evaluate the metric.

        :param self:
        :param predict: The prediction/inference `np.ndarray` results.
        :param labels_map: A map of other `np.ndarray` data fields,
            e.g. ground-truth labels, required for the metric evaluation.
        :return: Metric results. """
        pass

    @property
    @abc.abstractmethod
    def required_label_names(self) -> List[str]:
        """
        Returns the names of all required data fields.

        :return: A list of required data fields' name.
        """
        pass

    @abc.abstractmethod
    def metric_result(self):
        """

        :return: MetricResult object for this BaseMetric object.
        """
        pass

    @property
    def label_name(self):
        return self._label_name

    @label_name.setter
    def label_name(self, label_name: str):
        self._label_name = label_name

    def reconstruct(self, metric_value: float):
        metric_result = MetricResult(name=self._name,
                                     metric_name=self._name,
                                     result=metric_value,
                                     meta=None,
                                     optimize_mode=self._optimize_mode)

        return metric_result
