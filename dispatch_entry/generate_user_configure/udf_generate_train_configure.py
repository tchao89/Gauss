"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
"""
# --------------- this block just for test ---------------
from pipeline.local_pipeline.mapping import EnvironmentConfigure
from utils.bunch import Bunch
from utils.yaml_exec import yaml_write

user_feature = "/home/liangqian/Gauss/test_dataset/feature_conf.yaml"
environ_configure = EnvironmentConfigure(work_root="/home/liangqian/Gauss/experiments")

pipeline_dict = Bunch()
# ["udf", "auto", "multi_udf"]
pipeline_dict.mode = "udf"
# initial model path, optional: str or None, and it's different from increment model setting.
# This is used to train a better model instead of increment.
# if this value is not None, user can just use one model in value: model_zoo
pipeline_dict.init_model_root = None
# choose different supervised selector, optional: ["model_select", "topk_select"]
pipeline_dict.supervised_selector_mode = "topk_select"
# This value is used to set transform type in regression task, eg: {"target_name": "log"}
# ["log", "exp", "pow", None] is supported for now.
pipeline_dict.label_switch_type = None
# Because udf metric using in model evaluation may reduce bad results,
# this bool value is used to avoid this.
pipeline_dict.metric_eval_used_flag = False
pipeline_dict.work_root = environ_configure.work_root
# optional: ["binary_classification", "multiclass_classification", "regression"]
pipeline_dict.task_name = "binary_classification"
# optional: ["auc", "binary_f1", "multiclass_f1", "mse"]
# This value will decided the way auto ml component chooses the best model.
pipeline_dict.metric_name = "binary_f1"
# optional: ["mse", "binary_logloss", "None"]
# This value will customize the loss function of model, and it can be set None.
# if None, default loss will be chosen.
pipeline_dict.loss_name = None
# optional: ["libsvm", "txt", "csv"]
pipeline_dict.data_file_type = "csv"
# pipeline do not need to get target names in libsvm and txt file.
pipeline_dict.target_names = ["deposit"]
pipeline_dict.use_weight_flag = False
# weight_column_name is a string value, which means a specific column names weight_column_name in a csv file or last column in txt or libsvm
# using as sample weight. this value should be set "-1" if dataset file type is libsvm or txt.
pipeline_dict.weight_column_name = None
# format: {"label_name": {label_value: weight_value, ...}}, if no label value, choose target_A, target_B, ... instead.
# eg. {"target_A": {1: 1.9, -1: 1}}, {-1: {1: 1.9, -1: 1}}, {-2: {"yes": 1.9, "no": 1}}
# this interface will be reserved because anyone who is good at weight setting could use it conveniently
# this interface could be set False permanently if it doesn't need.
pipeline_dict.dataset_weight_dict = None
pipeline_dict.train_column_name_flag = True
pipeline_dict.train_data_path = "/home/liangqian/文档/公开数据集/bank/bank.csv"
pipeline_dict.val_column_name_flag = True
pipeline_dict.val_data_path = None
pipeline_dict.feature_configure_path = "/home/liangqian/文档/公开数据集/bank/bank_skip.yaml"
pipeline_dict.dataset_name = "plaindataset"
pipeline_dict.model_zoo = ["lightgbm"]
pipeline_dict.data_clear_flag = True
pipeline_dict.feature_generator_flag = True
pipeline_dict.unsupervised_feature_selector_flag = False
pipeline_dict.supervised_feature_selector_flag = False
user_config_path = environ_configure.work_root + "/train_user_config.yaml"
yaml_write(yaml_dict=dict(pipeline_dict), yaml_file=user_config_path)
system_config_path = "/home/liangqian/Gauss/configure_files/system_config/system_config.yaml"
print(system_config_path)
print(user_config_path)
