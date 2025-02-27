# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import abc

import tensorflow as tf


class BaseLoss(metaclass=abc.ABCMeta):
    """Base class for a loss function component.

    All subclasses of BaseLoss must override `loss_fn` method.
    """

    @abc.abstractmethod
    def loss_fn(self, logits: tf.Tensor, examples: tf.train.Example) -> tf.Tensor:
        """Build loss function.

        :param logits: The input logits to build the loss function.
        :param examples: The input `tf.Example` from which to abtain
            all required data fields.
        :return: A loss `tf.Tensor`. """
        pass
