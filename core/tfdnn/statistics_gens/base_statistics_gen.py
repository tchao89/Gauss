# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import division
from __future__ import absolute_import

import abc

from core.tfdnn.statistics_gens.statistics import Statistics


class BaseStatisticsGen(metaclass=abc.ABCMeta):
    """Base class for a statistics generator component.

    All subclasses of BaseStatisticsGen must override `run` method to generate
    feature statistics.
    """

    @abc.abstractmethod
    def run(self) -> Statistics:
        """Calculate feature statistics.

        :return: Feature statistics results."""
        pass
