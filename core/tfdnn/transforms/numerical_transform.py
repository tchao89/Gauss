# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from core.tfdnn.transforms.base_transform import BaseTransform


class ClsNumericalTransform(BaseTransform):

    def __init__(self, statistics, feature_names):
        self._statistics = statistics
        self._feature_names = feature_names

    def _transform_fn(self, example):
        for feature in self._feature_names:
            example[feature] = self._nonlinear_map_and_normalize(
                value=example[feature], 
                stat=self._statistics.stats[feature]
            )
        return example

    def _nonlinear_map_and_normalize(self, value, stat):
        value = tf.clip_by_value(tf.cast(value, tf.float32), stat.min, stat.max)
        linear_value = (value - stat.min) / (stat.max - stat.min + 1e-20)
        squared_value = tf.square(linear_value)
        square_root_value = tf.sqrt(linear_value)
        log_value = tf.math.divide(
            tf.math.log(value - stat.min + 1.0),
            tf.cast(tf.math.log(stat.max - stat.min + 1.0), tf.float32) + 1e-20,
        )

        return tf.concat([linear_value, squared_value, square_root_value, log_value],
                         axis=1)


class RegNumericalTransform(BaseTransform):

    def __init__(self, statistics, feature_names):
        self._statistics = statistics
        self._feature_names = feature_names

    def _transform_fn(self, example):
        for feature in self._feature_names:
            example[feature] = self._nonlinear_map_and_normalize(
                value=example[feature], 
                stat=self._statistics.stats[feature]
            )
        return example

    def _nonlinear_map_and_normalize(self, value, stat):
        value = tf.clip_by_value(tf.cast(value, tf.float32), stat.min, stat.max)
        linear_value = (value - stat.min) / (stat.max - stat.min + 1e-20)
        squared_value = tf.square(linear_value)
        square_root_value = tf.sqrt(linear_value)
        log_value = tf.math.divide(
            tf.math.log(value - stat.min + 1.0),
            tf.cast(tf.math.log(stat.max - stat.min + 1.0), tf.float32) + 1e-20,
        )

        return tf.concat([linear_value, squared_value, square_root_value, log_value],
                         axis=1) 