import pandas as pd
from gauss.type_inference.type_inference import TypeInference
from entity.plain_dataset import PlaintextDataset
import numpy as np
import copy

data = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank.csv")
str_coordinate = [1, 2, 3, 4]
column = copy.deepcopy(data['age'])
column.loc[str_coordinate] = column.iloc[str_coordinate].apply(lambda x: np.nan)

# data.loc[str_coordinate, column] = data[column].iloc[str_coordinate].apply(lambda x: np.nan)
# data[column] = data[column].astype('float64')

test = PlaintextDataset(name="dataset", task_type="test",
                        data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank.csv", target_name=["campaign"])

dataset = test
print(type(dataset))
type_infer = TypeInference(name='type_infer', task_name='classification', train_flag=True)
feature_conf = type_infer.dtype_inference(dataset=dataset)
type_infer.ftype_inference(dataset=dataset)

for item in feature_conf.feature_dict.items():
    print(item[0], item[1].name, item[1].index, item[1].dtype, item[1].ftype)
