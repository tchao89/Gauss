import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

df = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_with_string.csv")
df.drop(["deposit"], axis=1, inplace=True)
df.to_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_with_string_predict.csv", index=False)
