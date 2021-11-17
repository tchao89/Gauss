"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import pandas as pd
import numpy as np


def reduce_data(data_path: str = None, dataframe: pd.DataFrame = None, column_name_flag: bool = False):
    """
    This method is used to reduce internal memory used by a dataframe.
    :param column_name_flag: bool value, and if this value is True, the first row of csv is column names.
    This value is only worked if data_path is not None.
    :param data_path: file path of dataset path.
    :param dataframe: dataset.
    :return: conditional return.
    """
    assert (data_path is not None) ^ (dataframe is not None)
    assert isinstance(column_name_flag, bool), \
        "columns_name_flag should be bool value, but get {} instead.".format(column_name_flag)

    if dataframe is not None:
        df = dataframe
    else:
        if column_name_flag:
            df = pd.read_csv(data_path, parse_dates=True, keep_date_col=True, header=0)
        else:
            df = pd.read_csv(data_path, parse_dates=True, keep_date_col=True, header=None)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != "object" and col_type != "category":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    if data_path is not None:
        return df
