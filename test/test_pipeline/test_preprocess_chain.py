import datetime
import os
import random
import string
from pipeline.preprocess_chain import PreprocessRoute
from pipeline.core_chain import CoreRoute

from utils.bunch import Bunch


def id_generator(size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))


time = datetime.datetime.now()
experiment_id = datetime.datetime.strftime(time, '%Y%m%d-%H:%M--') + id_generator()

root = "/home/liangqian/PycharmProjects/Gauss/experiments"
experiment_path = os.path.join(root, experiment_id)
os.mkdir(experiment_path)
model_path = os.path.join(experiment_path, "model_path")
os.mkdir(model_path)

feature_dict = Bunch()
feature_dict.user_feature = "/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml"
feature_dict.type_inference_feature = os.path.join(experiment_path, "type_inference_feature.yaml")
feature_dict.data_clear_feature = os.path.join(experiment_path, "data_clear_feature.yaml")
feature_dict.feature_generator_feature = os.path.join(experiment_path, "feature_generator_feature.yaml")
feature_dict.unsupervised_feature = os.path.join(experiment_path, "unsupervised_feature.yaml")
feature_dict.label_encoding_path = os.path.join(experiment_path, "label_encoding_path")

chain = PreprocessRoute(name="chain",
                        feature_path_dict=feature_dict,
                        task_type="classification",
                        train_flag=True,
                        train_data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical.csv",
                        val_data_path=None,
                        test_data_path=None,
                        target_names=["deposit"],
                        dataset_name="plaindataset",
                        type_inference_name="typeinference",
                        data_clear_name="plaindataclear",
                        data_clear_flag=True,
                        feature_generator_name="featuretools",
                        feature_generator_flag=True,
                        feature_selector_name="unsupervised",
                        feature_selector_flag=True)

entity_dict = chain.run()
for item in entity_dict.keys():
    print(item)
    print(entity_dict[item].get_dataset().data)
    print(entity_dict[item].get_dataset().target)

core_chain = CoreRoute(name="core_chain",
                       train_flag=True,
                       model_name="lightgbm",
                       model_save_root=os.path.join(experiment_path, "model_path/"),
                       target_feature_configure_path=os.path.join(experiment_path, "target_feature_feature.yaml"),
                       pre_feature_configure_path=os.path.join(experiment_path, "unsupervised_feature.yaml"),
                       label_encoding_path=os.path.join(experiment_path, "label_encoding_path"),
                       model_type="tree_model",
                       metrics_name="auc",
                       task_type="classification",
                       feature_selector_name="supervised_selector",
                       feature_selector_flag=False,
                       auto_ml_type="auto_ml",
                       auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config",
                       selector_config_path="/home/liangqian/PycharmProjects/Gauss/configure_files/selector_config")

core_chain.run(**entity_dict)
chain = PreprocessRoute(name="chain",
                        feature_path_dict=feature_dict,
                        task_type="classification",
                        train_flag=False,
                        train_data_path=None,
                        val_data_path=None,
                        test_data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical.csv",
                        target_names=["deposit"],
                        dataset_name="plaindataset",
                        type_inference_name="typeinference",
                        data_clear_name="plaindataclear",
                        data_clear_flag=True,
                        feature_generator_name="featuretools",
                        feature_generator_flag=True,
                        feature_selector_name="unsupervised",
                        feature_selector_flag=True)
chain.run()
core_chain = CoreRoute(name="core_chain",
                       train_flag=False,
                       model_name="lightgbm",
                       model_save_root=os.path.join(experiment_path, "model_path/"),
                       target_feature_configure_path=os.path.join(experiment_path, "target_feature_feature.yaml"),
                       pre_feature_configure_path=os.path.join(experiment_path, "unsupervised_feature.yaml"),
                       label_encoding_path=os.path.join(experiment_path, "label_encoding_path"),
                       model_type="tree_model",
                       metrics_name="auc",
                       task_type="classification",
                       feature_selector_name="supervised_selector",
                       feature_selector_flag=False,
                       auto_ml_type="auto_ml",
                       auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config",
                       selector_config_path="/home/liangqian/PycharmProjects/Gauss/configure_files/selector_config")

core_chain.run(**entity_dict)
print("finished")
