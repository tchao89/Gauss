from entity.plaintext_dataset import PlaintextDataset
import pandas as pd
from sklearn.datasets import dump_svmlight_file

a = [[1], [2], [3]]
r = a.pop(1)
print(a)
print(r)
target_df = pd.DataFrame(a, columns=['a'])
test = PlaintextDataset(input_path="/home/liangqian/Gauss/bank_numerical.csv", target_name="deposit")
print(test.load_data().data)
# dump_svmlight_file(X=test.load_data().data, y=test.load_data().target, f="bank_numerical.libsvm")
test = PlaintextDataset(input_path="/home/liangqian/Gauss/bank_numerical.libsvm")
print(test.load_data().data)
test = PlaintextDataset(input_path="/home/liangqian/Gauss/bank_numerical.txt")
print(test.load_data().data)
