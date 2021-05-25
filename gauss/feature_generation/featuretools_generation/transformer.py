# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: Fintech Innovation Lab

import json
import yaml
import numpy as np
import pandas as pd
from gauss.feature_generation.feature_generation_base import TransformerBase
from gauss.feature_generation.featuretools_generation import featuretools as ft


class Transformer(metaclass=TransformerBase):

    def __init__(self, dataframe: pd.DataFrame, name: str = "feature generation", train_flag: bool = True):
        super().__init__(name, train_flag)
        self.ft_entity = ft.EntitySet(id="New Entity")
        self.type_dict = {}
        self.trans_primitives = []
        self.dataframe = dataframe
        self.column_name = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def fit(self, X: np.ndarray):
        return self

    def test(self):
        self._train_run()

    def _train_run(self):
        primitives = ft.list_primitives()
        self.trans_primitives = list(primitives[primitives['type'] == 'transform']['name'].values)
        with open('/home/liangqian/Gauss/transformer_config.json', 'r') as json_file:
            self.type_dict = json.load(json_file)
        print(self.type_dict)


df = pd.read_csv("/home/liangqian/Gauss/bank_numerical.csv")
obj = Transformer(dataframe=df)
obj.test()
