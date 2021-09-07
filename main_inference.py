# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
import argparse

from utils.common_component import yaml_read, yaml_write
from local_pipeline.inference import Inference
from utils.Logger import logger
from utils.bunch import Bunch


# test programming
pipeline_dict = yaml_read(yaml_file="/home/liangqian/PycharmProjects/Gauss/experiments/JlrcPl" + "/pipeline_config.yaml")
pipeline_dict["model_name"] = "lightgbm"
work_root = pipeline_dict["work_root"]
pipeline_dict.update(yaml_read(yaml_file=work_root + "/pipeline_config.yaml"))
pipeline_dict["test_data_path"] = "/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical_test_realdata.csv"
pipeline_dict["out_put_path"] = pipeline_dict["work_root"]

yaml_write(yaml_dict=pipeline_dict, yaml_file=work_root + "/inference_user_config.yaml")

def main(config=work_root + "/inference_user_config.yaml"):
    logger.info("Reading inference configuration files.")
    pipeline_configure = yaml_read(config)
    pipeline_configure = Bunch(**pipeline_configure)

    pipeline_configure.system_config_root = "/home/liangqian/PycharmProjects/Gauss/configure_files"
    pipeline_configure.auto_ml_path = pipeline_configure.system_config_root + "/" + "automl_params"
    pipeline_configure.selector_config_path = pipeline_configure.system_config_root + "/" + "selector_params"
    system_config = yaml_read(pipeline_configure.system_config_root + "/" + "system_config/system_config.yaml")
    system_config = Bunch(**system_config)

    pipeline_configure.update(system_config)

    inference = Inference(name="inference", **pipeline_configure)

    inference.offline_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("configure-file")
    parser.add_argument("-config", type=str,
                        help="config file")

    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
