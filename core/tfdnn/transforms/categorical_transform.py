# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from core.tfdnn.transforms.base_transform import BaseTransform


class CategoricalTransform(BaseTransform):

    def __init__(self,
                 statistics,
                 feature_names,
                 embed_size=128,
                 embed_shards=1,
                 default_num_oov_buckets=20,
                 map_num_oov_buckets={},
                 map_top_k_to_select:dict={},
                 map_shared_embedding:dict={},
                 scope_name="categorical_transform"):
        self._statistics = statistics
        self._feature_names = feature_names
        self._default_num_oov_buckets = default_num_oov_buckets
        self._map_num_oov_buckets = map_num_oov_buckets 
        self._map_top_k_to_select = map_top_k_to_select
        self._map_shared_embedding = map_shared_embedding
        self._embed_size = embed_size
        self._embed_shards = embed_shards
        self._scope_name = scope_name

    def __repr__(self):
        return "CategoricalTransform object processing {num} features with {dim} embeding sizes"\
            .format(num=len(self._feature_names), dim=self._embed_size)


    def _transform_fn(self, example):
        hash_sizes = self._calculate_hash_sizes()
        self._embedding_tables = self._create_embedding_tables(hash_sizes)
        example = self._embedding_lookup(example)
        return example

    def _embedding_lookup(self, example):
        for fea_name in self._feature_names:
            if fea_name not in self._map_shared_embedding:
                example[fea_name] = tf.nn.embedding_lookup(
                    params=self._embedding_tables[fea_name], 
                    ids=example[fea_name]
                )
            else:
                shared_fea_name = self._map_shared_embedding[fea_name]
                example[fea_name] = tf.nn.embedding_lookup(
                    params=self._embedding_tables[shared_fea_name], 
                    ids=example[fea_name]
                )
        return example

    def _calculate_hash_sizes(self):
        hash_sizes = {}
        for fea_name in self._feature_names:
            if fea_name in self._map_shared_embedding:
                assert self._map_shared_embedding[fea_name] in self._feature_names
            else:
                num_oov_buckets = (
                    self._map_num_oov_buckets[fea_name]
                    if fea_name in self._map_num_oov_buckets
                    else self._default_num_oov_buckets
                )
                top_k = (
                    self._map_top_k_to_select[fea_name]
                    if fea_name in self._map_top_k_to_select
                    else None
                )
                vocab = []
                if fea_name in self._statistics.stats:
                    vocab = self._statistics.stats[fea_name].values_top_k(top_k)
                else:
                    print(
                        "WARNING: feature [%s] not found in statistics, use empty."
                        % fea_name
                    )
                hash_sizes[fea_name] = len(vocab) + num_oov_buckets
        return hash_sizes

    def _create_embedding_tables(self, hash_sizes):
        embedding_tables = {}
        with tf.compat.v1.variable_scope(self._scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            for fea_name in self._feature_names:
                if fea_name in self._map_shared_embedding:
                    assert self._map_shared_embedding[fea_name] in self._feature_names
                else:
                    embedding_tables[fea_name] = tf.compat.v1.get_variable(
                        name=fea_name + "_embed",
                        shape=[hash_sizes[fea_name], self._embed_size],
                        partitioner=tf.fixed_size_partitioner(
                            self._embed_shards, axis=0
                        ) if self._embed_shards > 1 else None,
                    )
        return embedding_tables
