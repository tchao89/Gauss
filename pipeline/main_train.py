import argparse

from pipeline.auto_modeling_tree import AutoModelingTree
from pipeline.udf_modeling_tree import UdfModelingTree
from utils.common_component import yaml_read


def main(config="./config.yaml"):
    pipeline_configure = yaml_read(config)
    if pipeline_configure.mode == "Auto":
        auto_model_tree = AutoModelingTree("auto",
                                           pipeline_configure.work_root,
                                           pipeline_configure.task_type,
                                           pipeline_configure.metric_name,
                                           pipeline_configure.train_data_path,
                                           pipeline_configure.val_data_path,
                                           pipeline_configure.feature_configue_path,
                                           pipeline_configure.dataset_type,
                                           pipeline_configure.type_inference,
                                           pipeline_configure.data_clear,
                                           pipeline_configure.feature_generator,
                                           pipeline_configure.unsupervised_feature_selector,
                                           pipeline_configure.supervised_feature_selector,
                                           pipeline_configure.auto_ml)

        auto_model_tree.run()
    elif pipeline_configure.mode == "Udf":
        udf_model_tree = UdfModelingTree("udf",
                                         pipeline_configure.work_root,
                                         pipeline_configure.task_type,
                                         pipeline_configure.metric_name,
                                         pipeline_configure.label_name,
                                         pipeline_configure.train_data_path,
                                         pipeline_configure.val_data_path,
                                         pipeline_configure.feature_configue_path,
                                         pipeline_configure.dataset_type,
                                         pipeline_configure.type_inference,
                                         pipeline_configure.data_clear,
                                         pipeline_configure.data_clear_flag,
                                         pipeline_configure.feature_generator,
                                         pipeline_configure.feature_generator_flag,
                                         pipeline_configure.unsupervised_feature_selector,
                                         pipeline_configure.unsupervised_feature_selector_flag,
                                         pipeline_configure.supervised_feature_selector,
                                         pipeline_configure.supervised_feature_selector_flag,
                                         pipeline_configure.model_zoo,
                                         pipeline_configure.auto_ml)

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
