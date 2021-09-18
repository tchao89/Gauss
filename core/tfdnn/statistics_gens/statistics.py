# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import os
import pickle
import numpy as np

from collections import defaultdict



class NumericalStat(object):

    def __init__(self):
        self._min = float("inf")
        self._max = float("-inf")
        self._sum = 0.0
        self._square_sum = 0.0
        self._count = 0

    def update(self, values):
        self._min = min(np.min(values), self._min)
        self._max = max(np.max(values), self._max)
        self._sum += np.sum(values)
        self._square_sum += np.sum(values * values)
        self._count += values.size

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def mean(self):
        return self._sum / float(self._count)

    @property
    def std(self):
        return (self._square_sum / float(self._count) - self.mean * self.mean) ** 0.5

    @property
    def n_samples(self):
        return self._count

    def __repr__(self):
        return "stat(type=numerical, min=%f, max=%f, mean=%f, std=%f, n=%d)" % (
            self.min,
            self.max,
            self.mean,
            self.std,
            self.n_samples,
        )


class CategoricalStat(object):

    def __init__(self):
        self._values = defaultdict(int)
        self._count = 0

    def update(self, values):
        for value in values.flatten():
            self._values[value] += 1
        self._count += values.size

    def values_top_k(self, top_k=None):
        sorted_values = [
            k for k, _ in sorted(self._values.items(),
                                 key=lambda item: item[1],
                                 reverse=True)
        ]
        if top_k is None:
            return sorted_values
        else:
            return sorted_values[:top_k]

    @property
    def n_samples(self):
        return self._count

    @property
    def total_values(self):
        return len(self._values)

    def __repr__(self):
        return "stat(type=categorical, top-5-values=%s, #values=%d, n=%d)" % (
            self.values_top_k(5),
            self.total_values,
            self.n_samples,
        )


class Statistics(object):

    def __init__(self):
        self._stats = dict()

    def update(self, name, values, cate_fea, num_fea):
        if name in num_fea:
            if name not in self._stats:
                self._stats[name] = NumericalStat()
            self._stats[name].update(values)
        elif name in cate_fea:
            if name not in self._stats:
                self._stats[name] = CategoricalStat()
            self._stats[name].update(values)

    @property
    def stats(self):
        return self._stats

    def save_to_file(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self._stats, f)

    def load_from_file(self, filepath):
        with open(filepath, "rb") as f:
            self._stats = pickle.load(f)

    def __repr__(self):
        return '\n'.join(["%s: %s" % (name, stat) for name, stat in self.stats.items()])
