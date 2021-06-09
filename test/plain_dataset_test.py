from entity.plain_dataset import PlaintextDataset
from sklearn.datasets import dump_svmlight_file
import pandas as pd


test = PlaintextDataset(name="dataset", task_type="test", data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank.csv", target_name=["campaign"])
print(test.get_dataset().target)
print(test.get_column_size())
print(test.get_row_size())
print("test repr")
print(repr(test))
print(test.get_target_name())

# data = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/bank.csv")
# target = data["deposit"]
# dump_svmlight_file(X=data, y=target, f="/home/liangqian/PycharmProjects/Gauss/bank.libsvm")

test = PlaintextDataset(name="dataset", task_type="test", data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical.libsvm", target_name=["deposit"])

print(test.load_data().data)
print(type(test.load_data().data))
print(repr(test))
print(test.get_column_size())
print(test.get_row_size())
print(test.get_target_name())

test = PlaintextDataset(name="dataset", task_type="test", data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical.txt", target_name=["deposit"])
print(test.load_data().data)
print(type(test.load_data().data))
print(repr(test))
print(test.get_column_size())
print(test.get_row_size())
print(test.get_target_name())
