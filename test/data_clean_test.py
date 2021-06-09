from entity.plain_dataset import PlaintextDataset
from gauss.data_clear.plain_data_clear import PlainDataClear
import pandas as pd


df = pd.read_csv("/home/liangqian/PycharmProjects/Gauss/test_dataset/bank.csv")
print(df.shape)
test = PlaintextDataset(name="dataset", task_type="test",
                        data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank.csv", target_name=["campaign"])

test_clean = PlainDataClear(name='clean', train_flag=True, enable=True, model_name="tree_model", feature_configure_path='/home/liangqian/PycharmProjects/Gauss/test/final_configure.yaml', strategy_dict=None)
test_clean.run(dataset=test)

print(test.get_dataset().data.head())
