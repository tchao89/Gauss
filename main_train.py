# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
import argparse

from local_pipeline.auto_modeling_graph import AutoModelingGraph
from local_pipeline.udf_modeling_graph import UdfModelingGraph
from local_pipeline.multiprocess_udf_graph import MultiprocessUdfModelingGraph
from utils.common_component import yaml_read


def main(config=config_path):
    pipeline_configure = yaml_read(config)
    pipeline_configure = Bunch(**pipeline_configure)

    pipeline_configure.system_config_root = "/home/liangqian/PycharmProjects/Gauss/configure_files"
    pipeline_configure.auto_ml_path = pipeline_configure.system_config_root + "/" + "automl_params"
    pipeline_configure.selector_config_path = pipeline_configure.system_config_root + "/" + "selector_params"
    system_config = yaml_read(pipeline_configure.system_config_root + "/" + "system_config/system_config.yaml")
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
                                        selector_config_path=pipeline_configure.selector_config_path)

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
