# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
import argparse

from utils.common_component import yaml_read, yaml_write
from local_pipeline.inference import Inference
from utils.Logger import logger
from utils.bunch import Bunch


def main(config=work_root + "/inference_user_config.yaml"):
    logger.info("Reading inference configuration files.")
    pipeline_configure = yaml_read(config)
    pipeline_configure = Bunch(**pipeline_configure)

    inference = Inference(name="inference",
                          work_root=pipeline_configure["work_root"],
                          model_name=pipeline_configure["model_name"],
                          out_put_path=pipeline_configure["out_put_path"])

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
