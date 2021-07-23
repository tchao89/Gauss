# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
import argparse

from pipeline.auto_modeling_tree import AutoModelingTree
from pipeline.udf_modeling_tree import UdfModelingTree
from utils.common_component import yaml_read

# this block just for test
from pipeline.mapping import EnvironmentConfigure
from utils.bunch import Bunch
from utils.common_component import yaml_write
from utils.Logger import logger


def main(config=config_path):
    pipeline_configure = yaml_read(config)
    pipeline_configure = Bunch(**pipeline_configure)

    if pipeline_configure.mode == "auto":
        auto_model_tree = AutoModelingTree(name="auto",
                                           work_root=pipeline_configure.work_root,
                                           task_type=pipeline_configure.task_type,
                                           metric_name=pipeline_configure.metric_name,
                                           train_data_path=pipeline_configure.train_data_path,
                                           val_data_path=pipeline_configure.val_data_path,
                                           feature_configure_path=pipeline_configure.feature_configure_path,
                                           target_names=pipeline_configure.target_names,
                                           dataset_type=pipeline_configure.dataset_type,
                                           type_inference=pipeline_configure.type_inference,
                                           data_clear=pipeline_configure.data_clear,
                                           feature_generator=pipeline_configure.feature_generator,
                                           unsupervised_feature_selector=pipeline_configure.unsupervised_feature_selector,
                                           supervised_feature_selector=pipeline_configure.supervised_feature_selector,
                                           auto_ml=pipeline_configure.auto_ml)

        auto_model_tree.run()

    elif pipeline_configure.mode == "udf":
        udf_model_tree = UdfModelingTree(name="udf",
                                         work_root=pipeline_configure.work_root,
                                         task_type=pipeline_configure.task_type,
                                         metric_name=pipeline_configure.metric_name,
                                         target_names=pipeline_configure.target_names,
                                         train_data_path=pipeline_configure.train_data_path,
                                         val_data_path=pipeline_configure.val_data_path,
                                         feature_configure_path=pipeline_configure.feature_configure_path,
                                         dataset_type=pipeline_configure.dataset_type,
                                         type_inference=pipeline_configure.type_inference,
                                         data_clear=pipeline_configure.data_clear,
                                         data_clear_flag=pipeline_configure.data_clear_flag,
                                         feature_generator=pipeline_configure.feature_generator,
                                         feature_generator_flag=pipeline_configure.feature_generator_flag,
                                         unsupervised_feature_selector=pipeline_configure.unsupervised_feature_selector,
                                         unsupervised_feature_selector_flag=pipeline_configure.unsupervised_feature_selector_flag,
                                         supervised_feature_selector=pipeline_configure.supervised_feature_selector,
                                         supervised_feature_selector_flag=pipeline_configure.supervised_feature_selector_flag,
                                         model_zoo=pipeline_configure.model_zoo,
                                         auto_ml=pipeline_configure.auto_ml)

        udf_model_tree.run()


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
