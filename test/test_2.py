from multiprocessing import Process, shared_memory
import xgboost as xgb
import pandas as pd
import numpy as np


def f(i):
    print(type(shared_data))
    df = pd.DataFrame(shared_data, columns=shared_columns)
    label = df['deposit']
    df.drop(['deposit'], inplace=True, axis=1)

    train_data = [data.values, label.values.flatten()]
    d_train = xgb.DMatrix(data=train_data[0], label=train_data[1], silent=True)
    params = {'max_depth': 9, 'eta': 0.1, 'objective': 'logloss'}
    _model = xgb.train(params,
                       d_train,
                       num_boost_round=200,
                       verbose_eval=False)

    print("processing: ", i)
    return _model


if __name__ == '__main__':
    shared_memory_data = None
    shared_memory_columns = None
    try:
        jobs = []
        data = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_/full_new.csv")
        values = data.values
        columns = np.array(data.columns)

        shared_memory_data = shared_memory.SharedMemory(name="dataset_numpy", create=True, size=values.nbytes)
        shared_memory_columns = shared_memory.SharedMemory(name="columns", create=True, size=columns.nbytes)

        buffer = shared_memory_data.buf
        buffer_columns = shared_memory_columns.buf

        shared_data = np.ndarray(values.shape, dtype=values.dtype, buffer=buffer)
        shared_columns = np.ndarray(columns.shape, dtype=columns.dtype, buffer=buffer_columns)

        shared_data[:] = data[:]
        shared_columns[:] = columns[:]

        for i in range(500):
            p = Process(target=f, args=(i,))
            jobs.append(p)

        for job in jobs:
            job.start()
            job.join()
    finally:
        shared_memory_data.unlink()
        shared_memory_columns.unlink()

while not q.empty():
    time.sleep(1)
    print(q.get())    # prints "[42, None, 'hello']"
