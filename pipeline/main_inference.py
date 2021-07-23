# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
import argparse

from utils.common_component import yaml_read, yaml_write
from pipeline.inference import Inference


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
