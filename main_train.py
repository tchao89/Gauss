# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
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
                                         user_feature="/home/liangqian/Gauss/test_dataset/feature_conf.yaml")

pipeline_dict = Bunch()
# ["udf", "auto", "multi_udf"]
pipeline_dict.mode = "udf"
pipeline_dict.work_root = environ_configure.work_root
pipeline_dict.task_name = "classification"
pipeline_dict.metric_name = "auc"
# optional: ["libsvm", "txt", "csv"]
pipeline_dict.data_file_type = "libsvm"
pipeline_dict.train_data_path = "/home/liangqian/文档/公开数据集/w1a/w1a.libsvm"
pipeline_dict.val_data_path = "/home/liangqian/文档/公开数据集/w1a/w1a.t.libsvm"
# pipeline do not need to get target names in libsvm and txt file.
pipeline_dict.target_names = ["deposit"]
pipeline_dict.feature_configure_path = environ_configure.user_feature_path
pipeline_dict.dataset_name = "plaindataset"
pipeline_dict.model_zoo = ["lightgbm"]
pipeline_dict.data_clear_flag = False
pipeline_dict.feature_generator_flag = False
pipeline_dict.unsupervised_feature_selector_flag = False
pipeline_dict.supervised_feature_selector_flag = False
config_path = environ_configure.work_root + "/train_user_config.yaml"
yaml_write(yaml_dict=dict(pipeline_dict), yaml_file=config_path)
# --------------- test block end ---------------


def main(config=config_path):
    pipeline_configure = yaml_read(config)
    pipeline_configure = Bunch(**pipeline_configure)

    pipeline_configure.system_configure_root = "/home/liangqian/Gauss/configure_files"
    pipeline_configure.auto_ml_path = pipeline_configure.system_configure_root + "/" + "automl_params"
    pipeline_configure.selector_configure_path = pipeline_configure.system_configure_root + "/" + "selector_params"
    system_config = yaml_read(pipeline_configure.system_configure_root + "/" + "system_config/system_config.yaml")
    system_config = Bunch(**system_config)

    pipeline_configure.update(system_config)

    if pipeline_configure.mode == "auto":
        model_graph = AutoModelingGraph(name="auto",
                                        work_root=pipeline_configure.work_root,
                                        task_name=pipeline_configure.task_name,
                                        metric_name=pipeline_configure.metric_name,
                                        train_data_path=pipeline_configure.train_data_path,
                                        val_data_path=pipeline_configure.val_data_path,
                                        feature_configure_path=pipeline_configure.feature_configure_path,
                                        target_names=pipeline_configure.target_names,
                                        dataset_name=pipeline_configure.dataset_name,
                                        type_inference_name=pipeline_configure.type_inference_name,
                                        data_clear_name=pipeline_configure.data_clear_name,
                                        feature_generator_name=pipeline_configure.feature_generator_name,
                                        unsupervised_feature_selector_name=pipeline_configure.unsupervised_feature_selector_name,
                                        supervised_feature_selector_name=pipeline_configure.supervised_feature_selector_name,
                                        auto_ml_name=pipeline_configure.tabular_auto_ml,
                                        opt_model_names=pipeline_configure.opt_model_names,
                                        auto_ml_path=pipeline_configure.auto_ml_path,
                                        selector_configure_path=pipeline_configure.selector_config_path)

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
