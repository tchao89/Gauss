import sys
import tensorflow as tf
BASE_DIR = '/home/gzqq/developer/CITIC_PLATFORM/Gauss_nn/'
sys.path.append(BASE_DIR)

from entity.model.dnn_model import GaussNN
from entity.dataset.plain_dataset import PlaintextDataset
from gauss.auto_ml.tabular_auto_ml import TabularAutoML
from entity.metrics.udf_metric import NNAUC


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

    model_root = "./experiment/NN_exp"
    save_statistics_path = "./test_dataset/stat/stat.pkl"
    save_graph_path = "./test_dataset/ckpt/ckpt_epoch-10.meta"
    save_ckpt_dir = "./test_dataset/ckpt/"
    
    auto_ml_path = "./configure_files/automl_config/"


nn = GaussNN(name="dnn",
                model_path=None,
                model_root=CONFIGS.model_root,
                model_config_root="./",
                feature_config_root="./",
                task_type=CONFIGS.task_type,
                train_flag=True)

nn._feature_list = CONFIGS.selected_features
nn._categorical_features = CONFIGS.categorical_features
nn._numerical_features = CONFIGS.numerical_features

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
metrics = NNAUC(name="auc", label_name=CONFIGS.target_name)

entity = {
    "model": nn,
    "dataset": dataset_train,
    "val_dataset": dataset_val,
    "metrics": metrics
}

automl = TabularAutoML(
    name="automl",
    optimize_mode="minimize",
    auto_ml_path=CONFIGS.auto_ml_path,
    train_flag=True,
    enable=True,
    opt_model_names=["tpe", "anneal"],
    trail_num=10
)

automl.run(**entity)