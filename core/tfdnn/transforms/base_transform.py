# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import abc
import tensorflow as tf

from typing import Callable


class BaseTransform(metaclass=abc.ABCMeta):
    """Base class for a transform component.

    All subclasses of BaseTransform must override _transform_fn method that processes
    and transforms a `tf.Example` to another `tf.Example` depends on its dtype.
    """


    @property
    def transform_fn(self) -> Callable[[tf.train.Example], tf.train.Example]:
        """Returns the transform function.

        :return: A function to transform `tf.Example`."""
        return self._transform_fn

    @abc.abstractmethod
    def _transform_fn(self, example: tf.train.Example) -> tf.train.Example:
        """Transforms `tf.Example`.

        :param example: The input `tf.Example` to be transformed.
        :return: The transformed `tf.Example` as output."""
        return example
