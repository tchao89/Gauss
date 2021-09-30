# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import os
import time
import tensorflow as tf

from core.tfdnn.utils.earlystop import Earlystop
from core.tfdnn.utils.loggers import TrainLogger
from core.tfdnn.utils.loggers import ValidateLogger


class Trainer(object):
    """Activating initializing and the trainer.

    Parameters:
    ----------
    dataset: tf.data.Dataset, whole dataset for training the model.
    validate_step: int, steps between runing validation function through whole 
        dataset.
    optimizer_type: str, 
    earlystopping: bool, Wether the Early stop function activated. If True, `patience`
        and `delta` should be provided, otherwise could be None.
    patience: int, toralance of the epoch number when monitored metrics decreasing.
    delta: int, float, toralance of the increasing value monitored metrics value.
    """

    def __init__(self,
                 dataset,
                 transform_functions,
                 train_fn,
                 validate_steps,
                 log_steps,
                 learning_rate,
                 optimizer_type="lazy_adam",
                 train_epochs=100,
                 earlystopping=False,
                 patience=None,
                 delta=None,
                 evaluator=None,
                 save_checkpoints_dir=None,
                 restore_checkpoint_dir=None,
                 validate_at_start=False,
                 tensorboard_logdir=None):
        self._dataset = dataset
        self._transform_functions = transform_functions
        self._train_fn = train_fn
        self._train_epochs = train_epochs
        self._save_checkpoints_dir = save_checkpoints_dir
        self._restore_checkpoint_dir = restore_checkpoint_dir
        self._validate_steps = validate_steps
        self._log_steps = log_steps
        self._learning_rate = learning_rate
        self._optimizer_type = optimizer_type
        self._validate_at_start = validate_at_start
        self._evaluator = evaluator
        self._earlystopping = earlystopping

        if self._earlystopping:
            self._earlystop = Earlystop(patience, delta)
        self._valid_logger = ValidateLogger(tensorboard_logdir)
        self._train_logger = TrainLogger(self._log_steps, tensorboard_logdir)

    def run(self, sess=None):
        with tf.device(self._get_device_setter()):
            self._loss, self._train_op, self._optimizer = self._build_train_graph()
            self._sess = sess or self._create_session_and_init()
            self._run_train_loop()
            if sess is None:
                self._sess.close()

    def _get_device_setter(self):
        return None

    def _create_session_and_init(self):
        sess = tf.compat.v1.Session()
        tf.compat.v1.global_variables_initializer().run(session=sess)
        tf.compat.v1.tables_initializer().run(session=sess)
        if self._restore_checkpoint_dir:
            checkpoint_saver = tf.train.Saver(max_to_keep=None)
            checkpoint_saver.restore(sess, tf.train.latest_checkpoint(self._restore_checkpoint_dir))
        return sess

    def _build_train_graph(self):
        optimizer = self._create_optimizer()
        transform_fn = self._join_pipeline(self._transform_functions)
        loss = self._train_fn(transform_fn(self._dataset.next_batch))
        global_step = tf.compat.v1.train.get_or_create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)
        return loss, train_op, optimizer

    def _create_optimizer(self):
        if self._optimizer_type == "sgd":
            return tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
        elif self._optimizer_type == "adam":
            return tf.compat.v1.train.AdamOptimizer(learning_rate=self._learning_rate)
        elif self._optimizer_type == "lazy_adam":
            return tf.contrib.opt.LazyAdamOptimizer(learning_rate=self._learning_rate)
        elif self._optimizer_type == "nadam":
            return tf.contrib.opt.NadamOptimizer(learning_rate=self._learning_rate)
        else:
            raise NotImplementedError(
                "optimizer type %s is not supported." % self._optimizer_type
            )

    def _run_train_loop(self):
        if self._validate_at_start:
            self._validate(epoch=0, step=0)

        step = 0
        for epoch in range(self._train_epochs):
            avg_loss = 0
            counter = 0
            self._dataset.init(self._sess)
            while True:
                success, step_loss = self._train_step(epoch, step)
                avg_loss += step_loss
                counter += 1
                if not success:
                    avg_loss /= counter
                    break
                step += 1
                if step % self._validate_steps == 0:
                    eval_result = self._run_validate_loop(epoch=epoch, step=step)
            if self._earlystopping:
                self._earlystop(epoch, avg_loss)
                if self._earlystop.flag:
                    break
            self._save_checkpoint(epoch + 1)
            
    def _run_validate_loop(self, epoch, step):
        if self._evaluator:
            eval_results = self._evaluator.run(sess=self._sess)
            if self._valid_logger:
                self._valid_logger.log_info(eval_results, epoch=epoch, step=step)

    def _train_step(self, epoch, step):
        try:
            t_start = time.time()
            loss = self._sess.run([self._loss, self._train_op])[0]
            t_end = time.time()
            if self._train_logger:
                self._train_logger.log_info(loss=loss,
                                            time=t_end - t_start,
                                            size=self._dataset.batch_size,
                                            epoch=epoch,
                                            step=step + 1)
            return True, loss
        except tf.errors.OutOfRangeError:
            return False, 0

    def _save_checkpoint(self, step, prefix="ckpt_epoch"):
        if self._save_checkpoints_dir:
            checkpoint_saver = tf.compat.v1.train.Saver(max_to_keep=100)
            checkpoint_path = os.path.join(self._save_checkpoints_dir, prefix)
            checkpoint_saver.save(
                sess=self._sess, 
                save_path=checkpoint_path, 
                global_step=step, 
            )

    def _join_pipeline(self, map_functions):

        def joined_map_fn(example):
            for map_fn in map_functions:
                example = map_fn(example)
            return example

        return joined_map_fn
