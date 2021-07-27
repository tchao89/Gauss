
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_with_string.csv")
df_v = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_with_string_predict.csv")
df.drop(["deposit"], axis=1, inplace=True)
default_cat_impute_model = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
default_cat_impute_model.fit(df["marital"].values.reshape(-1, 1))
res = default_cat_impute_model.transform(df_v["marital"].values.reshape(-1, 1))
print(res[:10])
