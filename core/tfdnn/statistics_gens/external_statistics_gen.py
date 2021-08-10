# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

from core.tfdnn.statistics_gens.statistics import Statistics
from core.tfdnn.statistics_gens.base_statistics_gen import BaseStatisticsGen


class ExternalStatisticsGen(BaseStatisticsGen):

    def __init__(self, filepath: str):
        self._filepath = filepath

    def run(self) -> Statistics:
        statistics = Statistics()
        statistics.load_from_file(self._filepath)
        return statistics
