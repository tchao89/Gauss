# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

from collections import defaultdict

import numpy as np
import tensorflow as tf


class Evaluator(object):

    def __init__(self,
                 dataset,
                 transform_functions,
                 eval_fn,
                 metrics,
                 restore_checkpoint_path=None):
        self._dataset = dataset
        self._transform_functions = transform_functions
        self._eval_fn = eval_fn
        self._metrics = metrics
        self._restore_checkpoint_path = restore_checkpoint_path

        self._predict, self._labels = self._build_eval_graph()

    def run(self, sess=None):
        self._sess = sess or self._create_session_and_init()
        results = self._run_eval_loop()
        if sess is None:
            self._sess.close()
        return results

    def close(self):
        tf.reset_default_graph()

    def _create_session_and_init(self):
        sess = tf.Session()
        if self._restore_checkpoint_path:
            checkpoint_saver = tf.train.Saver(max_to_keep=None)
            checkpoint_saver.restore(sess, tf.train.latest_checkpoint(self._restore_checkpoint_path))
        else:
            tf.compat.v1.global_variables_initializer().run(session=sess)
        tf.compat.v1.tables_initializer().run(session=sess)
        return sess

    def _run_eval_loop(self):
        self._dataset.init(self._sess)
        predict, labels = [], defaultdict(list)
        while True:
            try:
                results = self._sess.run(
                    [self._predict]
                    + [self._labels[name] for name in sorted(self._labels.keys())]
                )
            except tf.errors.OutOfRangeError:
                break
            predict.append(results[0])
            for name, result in zip(sorted(self._labels.keys()), results[1:]):
                labels[name].append(result)
        predict = np.concatenate(predict, axis=0)
        for label_name in labels.keys():
            labels[label_name] = np.concatenate(labels[label_name], axis=0)

        metric_results = {}
        for metric_name, metric in self._metrics.items():
            result = metric.evaluate(predict, labels)
            metric_results[metric_name] = result
        return metric_results

    def _build_eval_graph(self):
        next_batch = self._dataset.next_batch
        transform_fn = self._join_pipeline(self._transform_functions)
        output = self._eval_fn(transform_fn(next_batch))
        labels = {name: next_batch[name] for name in self._all_required_label_names()}
        return output, labels

    def _all_required_label_names(self):
        required = set()
        for metric in self._metrics.values():
            required.update(metric.required_label_names[0])
        return list(required)

    def _join_pipeline(self, map_functions):
        def joined_map_fn(example):
            for map_fn in map_functions:
                example = map_fn(example)
            return example

        return joined_map_fn
