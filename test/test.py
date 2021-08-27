import gc
import time
import os
import psutil
from multiprocessing import shared_memory, Pool
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np


# def get_current_memory_gb() -> dict:
#
#     pid = os.getpid()
#     p = psutil.Process(pid)
#     info = p.memory_full_info()
#     memory_usage = info.uss / 1024. / 1024. / 1024. + info.swap / 1024. / 1024. / 1024. + info.pss / 1024. / 1024. / 1024.
#
#     return {"memory_usage": memory_usage, "pid": pid}

# def lr_f(i):
#     label = shared_data[:, -1]
#     data = shared_data[:, :-1]
#     _model = LogisticRegression(max_iter=1000)
#     _model.fit(X=data, y=label)
#     print("processing: ", i)
#     return _model
#
def xgb_f(i):
    label = shared_data[:, -1]
    data = shared_data[:, :-1]
    print(0)
    d_train = xgb.DMatrix(data=data, label=label.flatten(), silent=True, nthread=1)
    params = {'max_depth': 9, 'eta': 0.1, 'objective': 'binary:logistic'}
    print(d_train.get_label())
    print("1")
    _model = xgb.train(params,
                       d_train,
                       num_boost_round=200,
                       verbose_eval=False)

    print("processing: ", i)
    return _model


def f(params):
    shared_data_memory = shared_memory.SharedMemory(name=params[0])
    shared_data = np.ndarray(params[1], dtype=params[2], buffer=shared_data_memory.buf)
    print("time sleeping...")
    print(get_current_memory_gb())
    time.sleep(100)
    assert 1 == 0
    label = shared_data[:, -1]
    data = shared_data[:, :-1]
    # data = dataset[0]
    # label = dataset[1]

    train_data = [data, label.flatten()]
    lgb_train = lgb.Dataset(data=train_data[0], label=train_data[1], free_raw_data=False, silent=True)
    params = {
        "objective": "binary",

        "num_class": 1,

        "metric": ["auc", "l2"],

        "num_leaves": 32,

        "learning_rate": 0.01,

        "feature_fraction": 0.8,

        "bagging_fraction": 0.8,

        "bagging_freq": 5,

        "verbose": -1,

        "max_depth": 9,

        "nthread": -1,

        "lambda_l2": 0.8
    }
    _model = lgb.train(params,
                       lgb_train,
                       num_boost_round=20,
                       verbose_eval=False)

    shared_memory_data.close()
    shared_memory_columns.close()
    print("processing: ", params)
    return _model


if __name__ == '__main__':
    shared_memory_data = None
    shared_memory_columns = None
    try:
        jobs = []
        data = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_/full_new.csv")
        values = data.values
        columns = np.array(data.columns)
        # label = data['deposit'].values
        # data.drop(['deposit'], axis=1, inplace=True)
        # data = data.values

        shared_memory_data = shared_memory.SharedMemory(create=True, size=values.nbytes)
        shared_memory_columns = shared_memory.SharedMemory(create=True, size=columns.nbytes)

        buffer = shared_memory_data.buf
        buffer_columns = shared_memory_columns.buf

        shared_data = np.ndarray(values.shape, dtype=values.dtype, buffer=buffer)
        shared_columns = np.ndarray(columns.shape, dtype=columns.dtype, buffer=buffer_columns)

        data_name = shared_memory_data.name
        data_shape = values.shape
        data_dtype = values.dtype

        shared_data[:] = data[:]
        shared_columns[:] = columns[:]
        # data = shared_data[:, :-1]

        del data, values
        gc.collect()

        # label = shared_data[:, -1].reshape(-1, 1)
        # d_test = xgb.DMatrix(data=data, silent=True)
        with Pool(processes=4) as pool:
            res = pool.map(f, [(data_name, data_shape, data_dtype),
                               (data_name, data_shape, data_dtype),
                               (data_name, data_shape, data_dtype),
                               (data_name, data_shape, data_dtype)])

        # for obj in res:
        #     # print(type(model))
        #     print(obj.predict(data))
    finally:
        print("finished")
        shared_memory_data.unlink()
        shared_memory_columns.unlink()

    print("training finished...")
