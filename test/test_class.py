import pandas as pd
import lightgbm as lgb

df = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank.csv")

label = df['age']
data = df

dataset = lgb.Dataset(data=data, label=label)
print(dataset.get_label())
