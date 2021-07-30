# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
import argparse

from utils.common_component import yaml_read, yaml_write
from pipeline.inference import Inference


# test programming
pipeline_dict = yaml_read(yaml_file="/home/liangqian/PycharmProjects/Gauss/experiments/38wQJG/final_config.yaml")
best_root = pipeline_dict["best_root"]
work_root = pipeline_dict["work_root"]

pipeline_dict.update(yaml_read(yaml_file=best_root + "/pipeline/configure.yaml"))
pipeline_dict["test_data_path"] = "/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_with_string_predict.csv"
pipeline_dict["out_put_path"] = pipeline_dict["work_root"]

yaml_write(yaml_dict=pipeline_dict, yaml_file=work_root + "/inference_config.yaml")

def main(config=work_root + "/inference_config.yaml"):
    configure = yaml_read(config)
    inference = Inference(name="inference", work_root=pipeline_dict["work_root"], out_put_path=configure["out_put_path"])
    inference.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("configure-file")
    parser.add_argument("-config", type=str,
                        help="config file")

    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
