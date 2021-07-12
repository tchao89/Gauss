import argparse

from utils.common_component import yaml_read
from pipeline.inference import Inference


def main(config="./config.yaml"):
    configure = yaml_read(config)
    inference = Inference(name="inference", work_root=configure.work_root, out_put_path=configure.out_put_path)
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
