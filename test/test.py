import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical.csv")
df = shuffle(df)
print(type(df))
df.to_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical.csv")
