# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: luoqing
import abc
from externals.multiple import MultipleMeta
from gauss.component import Component


class TransformerBase(Component):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name: str, train_flag: bool = True):
        super().__init__(name, train_flag)

    def _train_run(self):
        pass

    def _inference_run(self):
        pass
