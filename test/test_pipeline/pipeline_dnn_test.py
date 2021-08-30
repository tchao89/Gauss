import sys
import tensorflow as tf
BASE_DIR = '/home/gzqq/developer/CITIC_PLATFORM/Gauss_nn/'
sys.path.append(BASE_DIR)

from pipeline.mapping import EnvironmentConfigure
from pipeline.preprocess_chain import PreprocessRoute
from pipeline.core_chain import CoreRoute
from pipeline.udf_modeling_tree import UdfModelingTree

class CONFIGS: 
    econfig = EnvironmentConfigure(work_root="./experiment/", user_feature="./test_dataset/feature_conf.yaml")
    econfig.feature_dict().user_feature = econfig.user_feature_path

    feature_dict = {"user_feature": "./test_dataset/feature_conf.yaml",
                    "type_inference_feature": econfig.work_root + "/feature/" + EnvironmentConfigure.feature_dict().type_inference_feature,
                    "data_clear_feature": econfig.work_root + "/feature/" + EnvironmentConfigure.feature_dict().data_clear_feature,
                    "feature_generator_feature": econfig.work_root + "/feature/" + EnvironmentConfigure.feature_dict().feature_generator_feature,
                    "unsupervised_feature": econfig.work_root + "/feature/" + EnvironmentConfigure.feature_dict().unsupervised_feature,
                    "supervised_feature": econfig.work_root + "/feature/" + EnvironmentConfigure.feature_dict().supervised_feature,
                    "label_encoding_path": econfig.work_root + "/feature/" + EnvironmentConfigure.feature_dict().label_encoding_path,
                    "impute_path": econfig.work_root + "/feature/" + EnvironmentConfigure.feature_dict().impute_path,
                    "final_feature_config": econfig.work_root + "/feature/" + EnvironmentConfigure.feature_dict().final_feature_config}

    train_data = "./test_dataset/bank_numerical_train_realdata.csv"
    val_data = "./test_dataset/bank_numerical_val_realdata.csv"

    target_names = ["deposit"]

    model_root = econfig.work_root + "/model/dnn/"
    model_save_dir = model_root + "model_save"
    model_config_dir = model_root + "model_config"
    feature_config_dir = model_root + "/feature_config" 

    final_feature_config_path = "./test/test_pipeline/final_feature_config.yaml"

    auto_ml_config_dir = "./configure_files/automl_config/"
    feature_selector_config_dir = "./configure_files/selector_config/"


pre_process_chain = PreprocessRoute(
    name="chain",
    feature_path_dict=CONFIGS.feature_dict,
    task_type="classification",
    train_flag=True,
    train_data_path=CONFIGS.train_data,
    val_data_path=CONFIGS.val_data,
    test_data_path=None,
    target_names=CONFIGS.target_names
    )

pre_process_chain.run()


CoreRoute(
    name="core",
    train_flag=True,
    model_name="dnn",
    model_save_root=CONFIGS.model_save_dir,
    model_config_root=CONFIGS.model_config_dir,
    feature_config_root=CONFIGS.feature_config_dir,
    target_feature_configure_path=CONFIGS.feature_dict["final_feature_config"],
    pre_feature_configure_path=CONFIGS.feature_dict["unsupervised_feature"],
    label_encoding_path=CONFIGS.feature_dict["label_encoding_path"],
    model_type="udf",
    metrics_name="nnauc",
    task_type="classification",
    feature_selector_name="feature_selector",
    feature_selector_flag=True,
    auto_ml_type="None",
    auto_ml_path=CONFIGS.auto_ml_config_dir,
    selector_config_path=CONFIGS.feature_selector_config_dir
    )