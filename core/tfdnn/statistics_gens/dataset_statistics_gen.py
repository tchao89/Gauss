# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from core.tfdnn.statistics_gens.statistics import Statistics
from core.tfdnn.statistics_gens.base_statistics_gen import BaseStatisticsGen


class DatasetStatisticsGen(BaseStatisticsGen):

    def __init__(self, 
                    dataset,
                    categorical_features,
                    numerical_features, 
                    num_batches=None):
        self._dataset = dataset
        self._categorical_features = categorical_features
        self._numerical_features = numerical_features
        self._num_batches = num_batches

    def run(self) -> Statistics:
        sess = tf.compat.v1.Session()
        self._dataset.init(sess)
        statistics = Statistics()
        n = 0
        while True:
            try:
                batch = sess.run(self._dataset.next_batch)
            except tf.errors.OutOfRangeError:
                break
            for name, values in batch.items():
                statistics.update(
                    name=name, 
                    values=values,
                    cate_fea=self._categorical_features,
                    num_fea=self._numerical_features
                    )
            n += 1
            if n % 1000 == 0:
                print("Statistics collected for %d batches ..." % n)
            if self._num_batches is not None and n >= self._num_batches:
                break
        sess.close()
        print("Statistics collected for %d batches in total." % n)
        return statistics
