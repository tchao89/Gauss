from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.datasets import load_svmlight_file
import pandas as pd

def load_libsvm(data_path):
    _data, _target = load_svmlight_file(data_path)
    _data = _data.toarray()

    return _data, _target


df = pd.read_csv("/home/liangqian/Gauss/experiments/nST8yE/result.csv")
# _, target = load_libsvm(data_path="/home/liangqian/文档/公开数据集/w1a/w1a.t")
target = pd.read_csv("/home/liangqian/文档/公开数据集/test/valid_label.csv")["label"]
y_true = [1 if item > 0 else 0 for item in target]
y_pred = [1 if item > 0.5 else 0 for item in df.values.flatten()]
y_score = df.values.flatten()
print(roc_auc_score(y_true=y_true, y_score=y_score))
print(f1_score(y_true=y_true, y_pred=y_pred))
print(classification_report(y_true=y_true, y_pred=y_pred))
