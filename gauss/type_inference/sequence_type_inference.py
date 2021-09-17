# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic Inc. All rights reserved.
# Authors: Lab

import re
import numpy as np
import pandas as pd

from entity.dataset.base_dataset import BaseDataset
from gauss.type_inference.base_type_inference import BaseTypeInference
from entity.feature_configuration.feature_config import (
    FeatureConf,
    FeatureItemConf
    )
from utils.Logger import logger
from utils.common_component import (
    yaml_read,
    yaml_write
)

class SequenceTypeInference(BaseTypeInference):
    """Inference sequence dataset data types and feature types.

    Features data type can be inferencied by it's statistic information. 
    """
    EPSILON = 0.00001
    THRESHOLD = 0.95

    STRING = "string"
    FLOAT = "float"
    FLOAT64 = "float64"
    INT = "int"
    INT64 = "int64"
    DATE = "datetime"
    OBJ = "object"

    CATE = "category"
    NUM = "numerical"

    REG = "regression"
    CLS = "classification"
    MUL = "multi"
    UNI = "unique"

    def __init__(self, **params):

        super(SequenceTypeInference, self).__init__(
            name=params["name"],
            train_flag=params["train_flag"],
            task_name=params["task_name"],
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
        # self.target_check(dataset=entity["dataset"])
        self._check_init_final_conf()
        self.save_config()

    def _predict_run(self, **entity):
        config = yaml_read(self._final_file_path)
        if not set(entity["dataset"].columns) == set(config):
            raise ValueError(
                "dataset features doesn't match configuration defined features."
                )


    def dtype_inference(self, dataset: BaseDataset):
        data = dataset.get_dataset().data
        data_types = data.dtypes
        
        for idx, col_name in enumerate(data):
            feature_item_config = FeatureItemConf(name=col_name, index=idx)

            if self.INT in str(data_types[idx]):
                feature_item_config.dtype = self.INT64

            elif self.FLOAT in str(data_types[idx]):
                if self._is_int(data.loc[:, col_name]):
                    feature_item_config.dtype = self.INT64
                    dataset.need_data_clean = True
                else:
                    feature_item_config.dtype = self.FLOAT64

            elif data_types[idx] == self.OBJ or self.CATE:
                float_counter, str_idx = self._count_and_index(
                    series=data.loc[:,col_name]
                    )

                if float_counter/data.shape[0] > self.THRESHOLD:
                    feature_item_config.dtype = self.FLOAT64
                    series = data.loc[:,col_name].copy()
                    series.iloc[str_idx] = series.iloc[str_idx].\
                        apply(lambda x: np.nan)
                    series = pd.to_numeric(series)

                    if self._is_int(series):
                        feature_item_config.dtype = self.INT64
                    dataset.need_data_clean = True
                else:
                    feature_item_config.dtype = self.STRING

            self.final_feature_config.add_item_type(
                column_name=col_name, 
                feature_item_conf=feature_item_config
                )
            self._string_column_selector(col_name)

    def _is_int(self, series):
        """Return True if error between original and force converted
        less than EPSILON, else False.
        """
        int_counter = 0
        for item in series:
            if (not np.isnan(item)) and (abs(item-int(item)<self.EPSILON)):
                int_counter += 1
        if int_counter + series.isna().sum() == series.shape[0]:
            return True
        else:
            return False
    
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
        init = self.init_feature_config
        if init is not None \
            and init.feature_dict.get(fea_name) \
            and init.feature_dict.get(fea_name).dtype == self.STRING:

            self.final_feature_config.feature_dict[fea_name].\
                dtype = self.STRING


    def ftype_inference(self, dataset: BaseDataset):
        data = dataset.get_dataset().data

        for col_name in data.columns:
            final_config = self.final_feature_config.feature_dict[col_name]
            if final_config.dtype == self.STRING:
                final_config.ftype = self.CATE
            elif final_config.dtype == self.FLOAT64:
                final_config.ftype = self.NUM
            else:
                unique_counter = len(pd.unique(data.loc[:,col_name]))
                sample_counter = data.loc[:,col_name].shape[0]
                unique_ratio = unique_counter / sample_counter
                if unique_ratio < self.THRESHOLD:
                    final_config.ftype = self.CATE
                else:
                    final_config.ftype = self.NUM 
        self._datetime_column_selector(feature_name=col_name, dataset=data)

    def _datetime_column_selector(
        self, 
        feature_name: str, 
        dataset: pd.DataFrame
        ):

        def _is_datetime(x):
            if re.search(r"(\d{4}.\d{1,2}.\d{1,2})", str(x)):
                return True
            else:
                return False

        if self.init_feature_config is not None:
            init_feature_dict = self.init_feature_config.feature_dict
 
            if init_feature_dict.get(feature_name) is not None \
                and init_feature_dict[feature_name].ftype == self.DATE \
                and init_feature_dict[feature_name].dtype == self.STRING:

                column_unique = list(set(dataset[feature_name]))
                if all(map(_is_datetime, column_unique)):
                    self.final_feature_config.feature_dict[feature_name].\
                        ftype = self.DATE


    def target_check(self, dataset: BaseDataset):
        pass
    #     for label_name in dataset.get_dataset().target_names:
            # label = dataset.get_dataset().labels.loc[:, label_name]
            # # self._target_dtype_check(dataset, label)
    
    def _target_dtype_check(self, dataset, label):
        if dataset.task_type == self.REG:
            if self.FLOAT not in str(label.dtype):
                raise ValueError(
                    "Invalid label type for `regression` task, except `float32`,"
                    "`float64`, `{dtype}` received.".format(
                        dtype=str(label.dtype)
                    )
                )
        elif dataset.task_type == self.CLS:
            if self.INT not in str(label.dtype):
                raise ValueError(
                    "Invalid label type for `classification` task, except `int32`,"
                    "`int64`, `{dtype}` received.".format(
                        dtype=str(label.dtype)
                    )
                )

    def _check_init_final_conf(self):
        if self.init_feature_config is not None: 
            final_feature_dict = self.final_feature_config.feature_dict
            
            for fea_name, fea_info in self.init_feature_config.feature_dict.items():
                if self.final_feature_config.feature_dict.get(fea_name):
                    exception = False

                    if fea_info.name and \
                        fea_info.name != final_feature_dict[fea_name].name:
                        logger.info(
                            fea_name + " feature's name is different between yaml \
                                file and type inference."
                        )
                        exception = True

                    if fea_info.index and \
                        fea_info.index != final_feature_dict[fea_name].index:
                        logger.info(
                            fea_name + " feature's index is different between yaml \
                                file and type inference."
                        )
                        exception = True

                    if fea_info.dtype and \
                        fea_info.dtype != final_feature_dict[fea_name].dtype:
                        logger.info(
                            fea_name + " feature's dtype is different between yaml \
                                file and type inference."
                        )
                        exception = True

                    if fea_info.ftype and \
                        fea_info.ftype != final_feature_dict[fea_name].ftype:
                        logger.info(
                            fea_name + " feature's ftype is different between yaml \
                                file and type inference."
                        )
                        exception = True

                    if not exception:
                        logger.info(
                            "Customized feature " + fea_name + " matches type inference."
                        )
                else:
                    logger.info(fea_name + " feature dose not exist in type inference.")
        else:
            logger.info("initialization configuration file didn't provide.")


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