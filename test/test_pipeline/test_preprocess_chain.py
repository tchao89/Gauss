import datetime
import os
import random
import string
from pipeline.preprocess_chain import PreprocessRoute

from utils.bunch import Bunch


def id_generator(size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))


time = datetime.datetime.now()
experiment_id = datetime.datetime.strftime(time, '%Y%m%d-%H:%M--') + id_generator()

root = "/home/liangqian/PycharmProjects/Gauss/experiments"
experiment_path = os.path.join(root, experiment_id)
os.mkdir(experiment_path)

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
chain.run()

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
print("finished")
