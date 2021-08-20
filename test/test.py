from multiprocessing import shared_memory, Pool
# from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import abc


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

class abs_function:
    def __init__(self, i):
        self.i = i

    @abc.abstractmethod
    def f(self):
        pass

class Function(abs_function):
    def __init__(self, i):
        super().__init__(i)
        self.model = None

    def f(self):
        label = shared_data[:, -1]
        data = shared_data[:, :-1]

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
        print("processing: ", self.i)
        self.model = _model
        return _model

def func(i):
    obj = Function(i)
    obj.f()
    return obj


if __name__ == '__main__':
    shared_memory_data = None
    shared_memory_columns = None
    try:
        jobs = []
        data = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical_train_realdata.csv")
        values = data.values
        columns = np.array(data.columns)

        shared_memory_data = shared_memory.SharedMemory(create=True, size=values.nbytes)
        shared_memory_columns = shared_memory.SharedMemory(create=True, size=columns.nbytes)

        buffer = shared_memory_data.buf
        buffer_columns = shared_memory_columns.buf

        shared_data = np.ndarray(values.shape, dtype=values.dtype, buffer=buffer)
        shared_columns = np.ndarray(columns.shape, dtype=columns.dtype, buffer=buffer_columns)

        shared_data[:] = data[:]
        shared_columns[:] = columns[:]
        data = shared_data[:, :-1]
        # label = shared_data[:, -1].reshape(-1, 1)
        # d_test = xgb.DMatrix(data=data, silent=True)
        with Pool(processes=4) as pool:
            res = pool.map(func, [0, 1, 2, 3])

        for obj in res:
            # print(type(model))
            print(obj.model.predict(data))
    finally:
        shared_memory_data.unlink()
        shared_memory_columns.unlink()

    print("training finished...")
