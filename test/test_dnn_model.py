import sys
BASE_DIR = '/home/gzqq/developer/CITIC_PLATFORM/Gauss_nn/'
sys.path.append(BASE_DIR)

from entity.dataset.plain_dataset import PlaintextDataset
from entity.model.dnn_model import GaussNN
from entity.metrics.udf_metric import NNAUC

import os
# os.system("rm -rf ./experiment/NN_exp")

class CONFIGS:
    train_set = "./test_dataset/bank_numerical_train_realdata.csv"
    val_set = "./test_dataset/bank_numerical_val_realdata.csv"
    test_set = "./test_dataset/bank_numerical_test_realdata.csv"
    task_type = "classification"
    target_name = ["deposit"]

    selected_features = ["age", "loan", "balance", "marital", "job", "education", "housing", "duration", 
                         "day", "month"]
    categorical_features = ["age", "loan", "marital", "job", "education", "housing", "day", "month"]
    numerical_features = ["balance", "duration"]

    embed_size = 8

    model_root = "./experiment/NN_exp"
    save_statistics_path = "./test_dataset/stat/stat.pkl"
    save_graph_path = "./test_dataset/ckpt/ckpt_epoch-10.meta"
    save_ckpt_dir = "./test_dataset/ckpt/"

    params = {
        "validate_steps": 300,
        "log_steps": 300,
        "batch_size": 32,
        "embed_size": 8,
        "hidden_sizes": [128, 64, 32],
        "validate_steps": 300,
        "log_steps": 300,
        "learning_rate": 0.01,
        "optimizer_type": "sgd",
        "train_epochs": 500
    }

dataset_train = PlaintextDataset(name="train_set",
                                data_path=CONFIGS.train_set,
                                task_type=CONFIGS.task_type,
                                target_name=CONFIGS.target_name,
                                memory_only=True
                                )
dataset_val = PlaintextDataset(name="val_set",
                                data_path=CONFIGS.val_set,
                                task_type=CONFIGS.task_type,
                                target_name=CONFIGS.target_name,
                                memory_only=True
                                )

nn = GaussNN(name="nn_model",
                model_path=None,
                model_root=CONFIGS.model_root,
                model_config_root="./",
                feature_config_root="./",
                task_type=CONFIGS.task_type,
                train_flag=True)

nn._feature_list = CONFIGS.selected_features
nn._categorical_features = CONFIGS.categorical_features
nn._numerical_features = CONFIGS.numerical_features
nn._model_params = CONFIGS.params

metrics = NNAUC(name="auc", label_name=CONFIGS.target_name)

nn.update_params(**CONFIGS.params)

entity = {
    "dataset": dataset_train,
    "val_dataset": dataset_val,
    "metrics": metrics
}
nn.train(**entity)

# print(nn.eval())

# nn.update_best_model()
# print(nn._best_model_params)
# print(nn._best_metrics_result)
# print(nn._best_feature_list)

# nn.set_best()
# print(nn._model_params)
# print(nn._feature_list)

# dataset_test = PlaintextDataset(name="test_set",
#                                 data_path=CONFIGS.test_set,
#                                 task_type=CONFIGS.task_type,
#                                 # target_name=CONFIGS.target_name,
#                                 memory_only=True
#                                 )

# test_entity = {
#     "val_dataset": dataset_test
# }
# print(nn.predict(**test_entity))



