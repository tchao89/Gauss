# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from core.tfdnn.networks.base_network import BaseNetwork


class MlpClsNetwork(BaseNetwork):

    def __init__(self,
                 categorical_features,
                 numerical_features,
                 task_name,
                 activation,
                 loss=None,
                 hidden_sizes=[1024, 512, 512],
                 scope_name="mlp_network"):
        self._categorical_features = categorical_features
        self._numerical_features = numerical_features
        self._task_name = task_name
        self._loss = loss
        self._hidden_sizes = hidden_sizes
        self._scope_name = scope_name
        self._activation = self._create_activation_func(activation)

    def _train_fn(self, example):
        with tf.compat.v1.variable_scope(self._scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            logits = self._build_graph(example)
            loss = self._loss.loss_fn(logits, example)
            return loss

    def _eval_fn(self, example):
        with tf.compat.v1.variable_scope(self._scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            logits = self._build_graph(example)
            outputs = tf.sigmoid(logits)
            return outputs

    def _serve_fn(self, example):
        return self._eval_fn(example)

    def _build_graph(self, inputs):
        categorical_part = tf.concat(
            [tf.squeeze(inputs[name], axis=1) for name in self._categorical_features],
            axis=1,
        )
        numerical_part = tf.concat(
            [inputs[name] for name in self._numerical_features], axis=1
        )
        hidden = tf.concat([categorical_part, numerical_part], axis=1)
        for i, size in enumerate(self._hidden_sizes):
            hidden = tf.layers.dense(
                inputs=hidden, units=size, 
                activation=self._activation,
                name="fc_" + str(i)
            )
        outputs = tf.layers.dense(inputs=hidden, units=1, name="logits")
        return outputs

    def _get_serve_inputs(self):
        return self._numerical_features + self._categorical_features
