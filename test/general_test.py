import core.lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

params = {"num_class": 1, "num_leaves": 32, "min_data_in_leaf": 20,
          "learning_rate": 0.01, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
          "max_depth": 9, "nthread": -1, "lambda_l2": 0.8, "metric": "auc"}

dataset = pd.read_csv("/home/liangqian/文档/公开数据集/bank/bank.csv")
label = [1 if item == "yes" else 0 for item in dataset["deposit"].values]
# label = np.random.random((len(label), ))

dataset.drop(["deposit"], axis=1, inplace=True)
train_set, test_set, train_label, test_label = train_test_split(dataset, label)

for col in ["job", "marital", "education", "housing", "loan", "month"]:
    train_set[col] = train_set[col].astype("category")
    test_set[col] = test_set[col].astype("category")

lgb_train = lgb.Dataset(data=train_set, label=train_label)
lgb_test = lgb.Dataset(data=test_set, label=test_label)

num_boost_round = 15
assert isinstance(lgb_train, lgb.Dataset)

params["objective"] = "binary"
obj_function = None
model = lgb.train(
    params=params,
    train_set=lgb_train,
    num_boost_round=num_boost_round,
    valid_sets=lgb_test,
    fobj=obj_function,
    verbose_eval=False,
)
