# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

class Predictor(object):

    def __init__(self,
                 dataset,
                 transform_functions,
                 eval_fn,
                 restore_checkpoint_dir=None):
        self._dataset = dataset
        self._transform_functions = transform_functions
        self._eval_fn = eval_fn
        self._restore_checkpoint_dir = restore_checkpoint_dir

        self._predict = self._build_predict_graph()

    def run(self, sess=None):
        self._sess = sess or self._create_session_and_init()
        results = self._run_predict_loop()
        if sess is None:
            self._sess.close()
        return results

    def close(self):
        tf.reset_default_graph()

    def _create_session_and_init(self):
        sess = tf.Session()
        if self._restore_checkpoint_dir:
            checkpoint_saver = tf.train.Saver(max_to_keep=None)
            checkpoint_saver.restore(sess, tf.train.latest_checkpoint(self._restore_checkpoint_dir))
        else:
            tf.compat.v1.global_variables_initializer().run(session=sess)
        tf.compat.v1.tables_initializer().run(session=sess)
        return sess

    def _run_predict_loop(self):
        self._dataset.init(self._sess)
        predict = []
        while True:
            try:
                results = self._sess.run(
                    self._predict
                )
            except tf.errors.OutOfRangeError:
                break
            results = np.where(results<0.5, 0, 1)
            predict.append(results)
        return predict

    def _build_predict_graph(self):
        next_batch = self._dataset.next_batch
        transform_fn = self._join_pipeline(self._transform_functions)
        outputs = self._eval_fn(transform_fn(next_batch))
        return outputs

    def _join_pipeline(self, map_functions):
        def joined_map_fn(example):
            for map_fn in map_functions:
                example = map_fn(example)
            return example

        return joined_map_fn