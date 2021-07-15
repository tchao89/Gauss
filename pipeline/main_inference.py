import argparse

from utils.common_component import yaml_read, yaml_write
from pipeline.inference import Inference

# test programming
from utils.bunch import Bunch

# random string "79Uvzp" is user directory.
pipeline_dict = Bunch()
pipeline_dict.work_root = "/home/liangqian/PycharmProjects/Gauss/experiments/79Uvzp"
pipeline_dict.task_type = "classification"
pipeline_dict.metric_name = "auc"
pipeline_dict.test_data_path = "/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical_predict.csv"
pipeline_dict.out_put_path = "/home/liangqian/PycharmProjects/Gauss/experiments/79Uvzp"

pipeline_dict.dataset_name = "plaindataset"
pipeline_dict.type_inference = "typeinference"
pipeline_dict.data_clear = "plaindataclear"
pipeline_dict.feature_generator = "featuretools"
pipeline_dict.unsupervised_feature_selector = "unsupervised"
pipeline_dict.supervised_feature_selector = "supervised"
pipeline_dict.auto_ml = "auto_ml"

pipeline_dict.best_root = yaml_read(yaml_file=pipeline_dict.work_root + "/final_config.yaml")["best_root"]

pipeline_flags = yaml_read(yaml_file=pipeline_dict.best_root + "/pipeline/configure.yaml")
pipeline_dict = dict(pipeline_dict)
pipeline_dict.update(dict(pipeline_flags))


def main(config=pipeline_dict["work_root"] + "/inference_config.yaml"):
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
