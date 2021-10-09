"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
"""
import argparse

from local_pipeline.singleprocess.auto_modeling_graph import AutoModelingGraph
from local_pipeline.singleprocess.udf_modeling_graph import UdfModelingGraph
from local_pipeline.multiprocess.multiprocess_udf_graph import MultiprocessUdfModelingGraph
from utils.yaml_exec import yaml_read

# --------------- this block just for test ---------------
from local_pipeline.pipeline_utils.mapping import EnvironmentConfigure
from utils.bunch import Bunch
from utils.yaml_exec import yaml_write
from utils.Logger import logger

user_feature = "/home/liangqian/Gauss/test_dataset/feature_conf.yaml"
environ_configure = EnvironmentConfigure(work_root="/home/liangqian/Gauss/experiments",
                                         user_feature=None)

pipeline_dict = Bunch()
# ["udf", "auto", "multi_udf"]
pipeline_dict.mode = "udf"
# initial model path, optional: str or None, and it's different from increment model setting.
# This is used to train a better model instead of increment.
pipeline_dict.init_model_path = None
# choose different supervised selector, optional: ["model_select", "topk_select"]
pipeline_dict.supervised_selector_mode = "topk_select"
# Because udf metric using in model evaluation may reduce bad results,
# this bool value is used to avoid this.
pipeline_dict.metric_eval_used_flag = False
# modify, 最外层
pipeline_dict.use_weight_flag = False
# if this value is true and type of dataset file is csv, "data_weight" column will be seen as weight.
pipeline_dict.weight_column_flag = False
pipeline_dict.work_root = environ_configure.work_root
# optional: ["binary_classification", "multiclass_classification", "regression"]
pipeline_dict.task_name = "binary_classification"
# optional: ["auc", "binary_f1", "multiclass_f1"]
# This value will decided the way auto ml component chooses the best model.
pipeline_dict.metric_name = "binary_f1"
# format: {"label_name": {label_value: weight_value, ...}}, if no label value, choose target_A, target_B, ... instead.
# eg. {"target_A": {1: 1.9, -1: 1}}
# this interface will be reserved because anyone who is good at weight setting could use it conveniently
# this interface could be set False permanently if it doesn't need.
pipeline_dict.dataset_weight = None
# optional: ["mse", "binary_logloss", "None"]
# This value will customize the loss function of model, and it can be set None.
# if None, default loss will be chosen.
pipeline_dict.loss_name = None
# optional: ["libsvm", "txt", "csv"]
pipeline_dict.data_file_type = "libsvm"
pipeline_dict.train_data_path = "/home/liangqian/文档/公开数据集/a9a/a9a"
pipeline_dict.val_data_path = "/home/liangqian/文档/公开数据集/a9a/a9a.t"
# pipeline do not need to get target names in libsvm and txt file.
pipeline_dict.target_names = None
pipeline_dict.feature_configure_path = environ_configure.user_feature_path
pipeline_dict.dataset_name = "plaindataset"
pipeline_dict.model_zoo = ["lightgbm"]
pipeline_dict.data_clear_flag = False
pipeline_dict.feature_generator_flag = False
pipeline_dict.unsupervised_feature_selector_flag = False
pipeline_dict.supervised_feature_selector_flag = True
config_path = environ_configure.work_root + "/train_user_config.yaml"
yaml_write(yaml_dict=dict(pipeline_dict), yaml_file=config_path)
# --------------- test block end ---------------


def main(config=config_path):
    pipeline_configure = yaml_read(config)
    pipeline_configure = Bunch(**pipeline_configure)

    pipeline_configure.system_configure_root = "/home/liangqian/Gauss/configure_files"
    pipeline_configure.auto_ml_path = pipeline_configure.system_configure_root + "/" + "automl_params"
    pipeline_configure.selector_configure_path = pipeline_configure.system_configure_root + "/" + "selector_params"
    pipeline_configure.improved_selector_configure_path = pipeline_configure.system_configure_root + "/" + "improved_selector_params"

    system_config = yaml_read(pipeline_configure.system_configure_root + "/" + "system_config/system_config.yaml")
    system_config = Bunch(**system_config)

    pipeline_configure.update(system_config)

    if pipeline_configure.mode == "auto":
        model_graph = AutoModelingGraph(name="auto", **pipeline_configure)

        model_graph.run()

    elif pipeline_configure.mode == "udf":
        model_graph = UdfModelingGraph(name="udf", **pipeline_configure)

        model_graph.run()

    elif pipeline_configure.mode == "multi_udf":

        pipeline_configure.dataset_name = "multiprocess_" + pipeline_dict.dataset_name
        pipeline_configure.model_zoo = ["multiprocess_" + model_name for model_name in pipeline_configure.model_zoo]

        model_graph = MultiprocessUdfModelingGraph(name="udf", **pipeline_configure)

        model_graph.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("configure-file")
    parser.add_argument("-config", type=str,
                        help="config file")

    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()

    logger.info(environ_configure.work_root)
