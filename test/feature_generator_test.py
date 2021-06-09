from entity.plain_dataset import PlaintextDataset
from entity.feature_config import FeatureConf
from gauss.type_inference.type_inference import TypeInference
from gauss.feature_generation.featuretools_generation import FeatureToolsGenerator

test = PlaintextDataset(name="dataset",
                        task_type="test",
                        data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank.csv",
                        target_name=["campaign"])

test_yaml = FeatureConf(name='test', file_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml")

type_ = TypeInference(name="type_inference",
                      train_flag=True,
                      task_name="classification",
                      source_file_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml")
type_.run(dataset=test)

generator = FeatureToolsGenerator(name="feature_generator", train_flag=True, enable=True,
                                  feature_config_path="/home/liangqian/PycharmProjects/Gauss/test/final_configure.yaml",
                                  label_encoding_configure_path="/home/liangqian/PycharmProjects/Gauss/test/final_configure.db")
generator.run(dataset=test)
print(test.get_dataset().data)
