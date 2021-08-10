# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

from core.tfdnn.networks.base_network import BaseNetwork


class NullNetwork(BaseNetwork):

    def __init__(self, input_as_output):
        self._input_as_output = input_as_output

    def _train_fn(self, example):
        raise NotImplementedError

    def _eval_fn(self, example):
        return example[self._input_as_output]

    def _serve_fn(self, example):
        return self._eval_fn(example)

    def _get_serve_inputs(self):
        return self._input_as_output
