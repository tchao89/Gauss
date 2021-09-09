# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Tencent Inc. All rights reserved.
# Authors: roygan

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from components.metrics.base_metric import BaseMetric
from components.metrics.base_metric import MetricResult


class PctrConsistency(BaseMetric):

    def __init__(self, score_name):
        self._score_name = score_name

    def eval(self, predict, labels_map):
        scores = labels_map[self._score_name]
        correct_num = 0
        incorrect_num = 0
        for i in range(predict.size):
            if abs(scores[i] - predict[i]) < 1e-4:
                correct_num += 1
            else:
                incorrect_num += 1
                if incorrect_num < 10:
                    for key, value in labels_map.items():
                        print(key, ":", value[i])
                    print("predict: %f, label: %f" % (predict[i], scores[i]))
        consistency = correct_num / (float(predict.size) + 1e-20)
        return MetricResult(result=consistency, meta={'#': predict.size})

    @property
    def required_label_names(self):
        return [self._score_name]
