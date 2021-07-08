from gauss_factory.gauss_factory_producer import GaussFactoryProducer
from utils.bunch import Bunch

gauss_factory = GaussFactoryProducer()
entity_factory = gauss_factory.get_factory(choice="entity")
component_factory = gauss_factory.get_factory(choice="component")

# 创建plaindataset对象
dataset_params = Bunch(name="test", task_type="classification", data_pair=None,
                       data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical.csv",
                       target_name=["deposit"], memory_only=True)
plain_dataset = entity_factory.get_entity(entity_name="plaindataset", **dataset_params)
plain_dataset_test = entity_factory.get_entity(entity_name="plaindataset", **dataset_params)
dataset = Bunch(dataset=plain_dataset)
dataset_test = Bunch(dataset=plain_dataset_test)

# 创建feature_config对象
conf_params = Bunch(name="test", file_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml")
feature_conf = entity_factory.get_entity(entity_name="featureconfigure", **conf_params)
feature_conf.parse()

# 创建type_inference对象
inference_params = Bunch(name="test", task_name='classification', train_flag=True,
                         source_file_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml",
                         final_file_path="/home/liangqian/PycharmProjects/Gauss/test/factory_test/final_configure.yaml",
                         final_file_prefix="final")
type_infer = component_factory.get_component(component_name="typeinference", **inference_params)
# print(type(type_infer))
type_infer.run(**dataset)

# 数据清洗组件
clear_params = Bunch(name='clean', train_flag=True, enable=True, model_name="tree_model",
                     final_file_path="/home/liangqian/PycharmProjects/Gauss/test/factory_test/final_configure.yaml",
                     feature_configure_path="/home/liangqian/PycharmProjects/Gauss/test/factory_test/final_configure.yaml",
                     strategy_dict=None)
data_clear = component_factory.get_component(component_name="plaindataclear", **clear_params)
data_clear.run(**dataset)

# 特征生成组件
generation_params = Bunch(name="feature_generator", train_flag=True, enable=True,
                          final_file_path="/home/liangqian/PycharmProjects/Gauss/test/factory_test/final_configure.yaml",
                          feature_config_path="/home/liangqian/PycharmProjects/Gauss/test/factory_test/final_configure.yaml",
                          label_encoding_configure_path="/home/liangqian/PycharmProjects/Gauss/configure_files/final_configure.db")
feature_generation = component_factory.get_component(component_name="featuretoolsgeneration", **generation_params)
feature_generation.run(**dataset)
# print(dataset.dataset.get_dataset().data)
# print(dataset.dataset.get_dataset().feature_names)
# print(dataset.dataset.get_dataset().generated_feature_names)

# 无监督特征选择
u_params = Bunch(name="test", train_flag=True, enable=True,
                 feature_config_path="/home/liangqian/PycharmProjects/Gauss/test/factory_test/final_configure.yaml",
                 final_file_path="/home/liangqian/PycharmProjects/Gauss/test/factory_test/final_configure.yaml",
                 label_encoding_configure_path="/home/liangqian/PycharmProjects/Gauss/configure_files/final_configure.db",
                 feature_select_configure_path='/home/liangqian/PycharmProjects/Gauss/configure_files/unsupervised_selector_result.json')
u_selector = component_factory.get_component(component_name="unsupervisedfeatureselector", **u_params)
u_selector.run(**dataset)

print(dataset.dataset.get_dataset().data)
print(dataset.dataset.get_dataset().feature_names)
print(dataset.dataset.get_dataset().generated_feature_names)

# 监督特征选择
s_params = Bunch(name="test", train_flag=True, enable=True, metrics_name="AUC", task_name="classification",
                 feature_config_path="/home/liangqian/PycharmProjects/Gauss/configure_files/final_configure.yaml",
                 label_encoding_configure_path="/home/liangqian/PycharmProjects/Gauss/configure_files/final_configure.db",
                 selector_config_path='/home/liangqian/PycharmProjects/Gauss/configure_files/selector_config',
                 model_name="lightgbm", auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config",
                 model_save_path="/home/liangqian/PycharmProjects/Gauss/configure_files/model")
s_selector = component_factory.get_component(component_name="supervisedfeatureselector", **s_params)
s_selector.run(**dataset)

# 模型选择
# model_params = Bunch()
# model = entity_factory.get_entity(entity_name="lightgbm", **model_params)

# 评估指标选择
metric_params = Bunch()
# 自动机器学习对象

print("Finished.")
