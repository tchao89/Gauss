# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from core.tfdnn.losses.base_loss import BaseLoss


class MSELoss(BaseLoss):

    def __init__(self, label_name):
        self._label_name = label_name[0]

    def loss_fn(self, logits, examples):
        labels = tf.cast(examples[self._label_name], tf.float32)
        return self._mse_loss(logits, labels)

    def _mse_loss(self, logits, labels):
        sample_loss = tf.losses.mean_squared_error(
            labels=labels, predictions=logits
        )
        avg_loss = tf.reduce_mean(sample_loss)
        return avg_loss