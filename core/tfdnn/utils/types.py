# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import tensorflow as tf


class FeatureConfig(object):

    def __init__(self, name, dtype, size, default_value=None):
        assert dtype in ("int64", "float64", "string")
        self.name = name
        self.dtype = {"int64": tf.int64,
                      "float64": tf.float32,
                      "string": tf.string}[dtype]
        self.size = size
        if default_value is None:
            if dtype == "string":
                self.default_value = "UNK"
            else:
                self.default_value = 0
        else:
            self.default_value = default_value

    def __repr__(self):
        return "FeatureConfig: {{name: {}, dtype: {}, size: {}}}".format(self.name, self.dtype, self.size)