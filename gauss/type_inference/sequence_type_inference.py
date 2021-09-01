# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic Inc. All rights reserved.
# Authors: Lab

import numpy as np
import pandas as pd

from entity.dataset.tf_sequence_dataset import SequenceDataset
from gauss.type_inference.base_type_inference import BaseTypeInference
from entity.feature_configuration.feature_config import (
    FeatureConf,
    FeatureItemConf
    )
from utils.common_component import (
    yaml_read,
    yaml_write
)
from utils.utils import CONST

const = CONST()

EPSILON = const("EPSILON", 0.00001)
THRESHOLD = const("THRESHOLD", 0.95)

STRING = const("STRING", "string")
FLOAT = const("FLOAT", "float")
FLOAT64 = const("FLOAT64", "float64")
INT = const("INT", "int")
INT64 = const("INT64", "int64")
DATE = const("DATE", "datetime")
OBJ = const("OBJ", "object")

CATE = const("CATE", "category")
NUM = const("NUM", "numerical")


class SequenceTypeInference(BaseTypeInference):
    
    def __init__(self, **params):

        super(SequenceTypeInference, self).__init__(
            name=params["name"],
            train_flag=params["train_flag"],
            source_file_path=params["source_file_path"],
            final_file_path=params["final_file_path"]
        )

        if self._source_file_path is not None:
            self.init_feature_config = FeatureConf(
                name="source_feature_config",
                file_path=self._source_file_path
            )
            self.init_feature_config.parse()
        else:
            self.init_feature_config = None

        self.final_feature_config = FeatureConf(
            name="gened_feature_config",
            file_path=self._final_file_path
            )

    def _train_run(self, **entity):
        self.dtype_inference(dataset=entity["dataset"])
        self.ftype_inference(dataset=entity["dataset"])

    def _predict_run(self, **entity):
        config = yaml_read(self._final_file_path)
        if not set(entity["dataset"].columns) == set(config):
            raise ValueError("dataset features doesn't match configuration defined features.")

    def dtype_inference(self, dataset: SequenceDataset):
        data = dataset.get_dataset().data
        data_types = data.dtypes
        
        for idx, col_name in enumerate(data):
            feature_item_config = FeatureItemConf(name=col_name, index=idx)

            if INT in str(data_types[idx]):
                feature_item_config.dtype = INT64

            elif FLOAT in str(data_types[idx]):
                if self._is_int(data.loc[:, col_name]):
                    feature_item_config.dtype = INT64
                    dataset.need_data_clean = True
                else:
                    feature_item_config.dtype = FLOAT64

            elif data_types[idx] == OBJ or CATE:
                float_counter, str_idx = self._count_and_index(data.loc[:,col_name])

                if float_counter/data.shape[0] > THRESHOLD:
                    feature_item_config.dtype = FLOAT64
                    series = data.loc[:,col_name].copy()
                    series.iloc[str_idx] = series.iloc[str_idx].apply(lambda x: np.nan)
                    series = pd.to_numeric(series)

                    if self._is_int(series):
                        feature_item_config.dtype = INT64
                    dataset.need_data_clean = True
                else:
                    feature_item_config.dtype = STRING

            self.final_feature_config.add_item_type(
                column_name=col_name, 
                feature_item_conf=feature_item_config
                )
            self._string_column_selector(col_name)

    def _is_int(self, series):
        int_counter = 0
        for item in series:
            if (not np.isnan(item)) and (abs(item-int(item)<EPSILON)):
                int_counter += 1
        if int_counter + series.isna().sum() == series.shape[0]:
            return True
    
    def _count_and_index(self, series):
        float_counter = 0
        string_idx = []
        for item in series:
            try:
                float(item)
                float_counter += 1
            except ValueError:
                string_idx.append(string_idx)
        return float_counter, string_idx

    def _string_column_selector(self, fea_name: str):
        if self.init_feature_config and self.init_feature_config.feature_dict.get(fea_name) \
            and self.init_feature_config.feature_dict.get(fea_name).dtype == STRING:

            self.final_feature_config.feature_dict[fea_name].dtype = STRING

    def ftype_inference(self, dataset: SequenceDataset):
        data = dataset.get_dataset().data

        for col_name in data.columns:
            final_config = self.final_feature_config.feature_dict[col_name]
            if final_config.dtype == STRING:
                final_config.ftype = CATE
            elif final_config.dtype == FLOAT64:
                final_config.ftype = NUM
            else:
                unique_counter = len(pd.unique(data.loc[:,col_name]))
                sample_counter = data.loc[:,col_name].shape[0]
                unique_ratio = unique_counter / sample_counter
                if unique_ratio < THRESHOLD:
                    final_config.ftype = CATE
                else:
                    final_config.ftype = NUM
            
    def target_check(self):
        pass

    def save_config(self):
        yaml = {}
        for fea_name, info in self.final_feature_config.feature_dict.items():
            item = {
                "name": info.name, 
                "index": info.index, 
                "dtype": info.dtype,
                "ftype": info.ftype,
                "size": info.size
                }
            yaml[fea_name] = item
            
        yaml_write(yaml_dict=yaml, yaml_file=self._final_file_path)