"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic Inc. All rights reserved.
Authors: Lab
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np

from entity.entity import Entity


class LossResult(Entity):
    """Class for loss result."""
    def __init__(self, name: str, result: dict):
        """Construct a loss result.
        :param result: The loss' result.
        """
        assert "loss" in result, "Key: loss must be in result."
        assert "grad" in result, "Key: grad must be in result."
        assert "hess" in result, "Key: hess must be in result."

        self._result = result

        self._loss = result.get("loss")
        self._grad = result.get("grad")
        self._hess = result.get("hess")

        super(LossResult, self).__init__(
            name=name,
        )

    @property
    def result(self):
        return self._result

    @property
    def loss(self):
        return self._loss

    @property
    def grad(self):
        return self._grad

    @property
    def hess(self):
        return self._hess

    def __repr__(self):
        if self._result is not None:
            return str(self._result)
        return None

    def __cmp__(self, other):
        return - self._loss + other.loss

class BaseLoss(Entity):
    """Base class for a loss function component.

    All subclasses of BaseLoss must override `loss_fn` method.
    """
    def __init__(self, name: str):
        super().__init__(name=name)

    @abc.abstractmethod
    def loss_fn(self, logits: np.ndarray, examples: np.ndarray) -> LossResult:
        """Build loss function.

        :param logits: The input logits to build the loss function.
        :param examples: The input `tf.Example` from which to obtain
            all required data fields.
        :return: A loss `np.ndarray`. """
