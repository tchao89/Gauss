import os
from pipeline.preprocess_chain import PreprocessRoute
from pipeline.core_chain import CoreRoute
from pipeline.mapping import EnvironmentConfigure


environ_configure = EnvironmentConfigure(env="test")
experiment_path, feature_dict = environ_configure.env_conf()

chain = PreprocessRoute(name="chain",
                        feature_path_dict=feature_dict,
                        task_type="classification",
                        train_flag=True,
                        train_data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical.csv",
                        val_data_path=None,
                        test_data_path=None,
                        target_names=["deposit"],
                        dataset_name="plaindataset",
                        type_inference_name="typeinference",
                        data_clear_name="plaindataclear",
                        data_clear_flag=True,
                        feature_generator_name="featuretools",
                        feature_generator_flag=True,
                        feature_selector_name="unsupervised",
                        feature_selector_flag=True)

entity_dict = chain.run()
for item in entity_dict.keys():
    print(item)
    print(entity_dict[item].get_dataset().data)
    print(entity_dict[item].get_dataset().target)

core_chain = CoreRoute(name="core_chain",
                       train_flag=True,
                       model_name="lightgbm",
                       model_save_root=os.path.join(experiment_path, "model_path/"),
                       target_feature_configure_path=os.path.join(experiment_path, "target_feature_feature.yaml"),
                       pre_feature_configure_path=os.path.join(experiment_path, "unsupervised_feature.yaml"),
                       label_encoding_path=os.path.join(experiment_path, "label_encoding_path"),
                       model_type="tree_model",
                       metrics_name="auc",
                       task_type="classification",
                       feature_selector_name="supervised_selector",
                       feature_selector_flag=False,
                       auto_ml_type="auto_ml",
                       auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config",
                       selector_config_path="/home/liangqian/PycharmProjects/Gauss/configure_files/selector_config")

core_chain.run(**entity_dict)
chain = PreprocessRoute(name="chain",
                        feature_path_dict=feature_dict,
                        task_type="classification",
                        train_flag=False,
                        train_data_path=None,
                        val_data_path=None,
                        test_data_path="/home/liangqian/PycharmProjects/Gauss/test_dataset/bank_numerical.csv",
                        target_names=["deposit"],
                        dataset_name="plaindataset",
                        type_inference_name="typeinference",
                        data_clear_name="plaindataclear",
                        data_clear_flag=True,
                        feature_generator_name="featuretools",
                        feature_generator_flag=True,
                        feature_selector_name="unsupervised",
                        feature_selector_flag=True)
chain.run()
core_chain = CoreRoute(name="core_chain",
                       train_flag=False,
                       model_name="lightgbm",
                       model_save_root=os.path.join(experiment_path, "model_path/"),
                       target_feature_configure_path=os.path.join(experiment_path, "target_feature_feature.yaml"),
                       pre_feature_configure_path=os.path.join(experiment_path, "unsupervised_feature.yaml"),
                       label_encoding_path=os.path.join(experiment_path, "label_encoding_path"),
                       model_type="tree_model",
                       metrics_name="auc",
                       task_type="classification",
                       feature_selector_name="supervised_selector",
                       feature_selector_flag=False,
                       auto_ml_type="auto_ml",
                       auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config",
                       selector_config_path="/home/liangqian/PycharmProjects/Gauss/configure_files/selector_config")

core_chain.run(**entity_dict)
print("finished")
