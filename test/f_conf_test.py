from entity.feature_config import FeatureConf
from gauss.type_inference.type_inference import TypeInference
from entity.plain_dataset import PlaintextDataset


test = PlaintextDataset(name="dataset", task_type="test", data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank.csv", target_name=["campaign"])

test_yaml = FeatureConf(name='test', file_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml")

type_infer = TypeInference(name='type infer',
                           task_name='classification',
                           train_flag=True,
                           source_file_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml")

type_infer.run(dataset=test)
