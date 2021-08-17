# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import pandas as pd
import numpy as np


def check_data(already_data_clear, model_name):
    assert isinstance(already_data_clear, bool)
    assert isinstance(model_name, str)

    if already_data_clear is not True:
        if model_name not in ["lightgbm", "xgboost", "catboost"]:
            return False
    return True


def reduce_data(data_path: str = None, dataframe: pd.DataFrame = None):
    assert (data_path is not None) ^ (dataframe is not None)

    if dataframe is not None:
        df = dataframe
    else:
        df = pd.read_csv(data_path, parse_dates=True, keep_date_col=True)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:

            df[col] = df[col].astype('category')

    if data_path is not None:
        return df


def safe_run_component():
    pass
