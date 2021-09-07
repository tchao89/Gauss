from local_pipeline.preprocess_chain import PreprocessRoute
from local_pipeline.core_chain import CoreRoute
from local_pipeline.mapping import EnvironmentConfigure


user_feature = "/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml"
environ_configure = EnvironmentConfigure(work_root="/home/liangqian/PycharmProjects/Gauss/experiments",
                                         user_feature="/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml")
print(environ_configure.file_path)

chain = PreprocessRoute(name="chain",
                        feature_path_dict=environ_configure.file_path,
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
                       model_save_root=environ_configure.model_save_path,
                       target_feature_configure_path=environ_configure.file_path.supervised_feature,
                       pre_feature_configure_path=environ_configure.file_path.unsupervised_feature,
                       label_encoding_path=environ_configure.file_path.label_encoding_path,
                       model_type="tree_model",
                       metrics_name="auc",
                       task_type="classification",
                       feature_selector_name="supervised_selector",
                       feature_selector_flag=True,
                       auto_ml_type="auto_ml",
                       auto_ml_path="/configure_files/automl_params",
                       selector_config_path="/configure_files/selector_params")

core_chain.run(**entity_dict)
chain = PreprocessRoute(name="chain",
                        feature_path_dict=environ_configure.file_path,
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
                       model_save_root=environ_configure.model_save_path,
                       target_feature_configure_path=environ_configure.file_path.supervised_feature,
                       pre_feature_configure_path=environ_configure.file_path.unsupervised_feature,
                       label_encoding_path=environ_configure.file_path.label_encoding_path,
                       model_type="tree_model",
                       metrics_name="auc",
                       task_type="classification",
                       feature_selector_name="supervised_selector",
                       feature_selector_flag=True,
                       auto_ml_type="auto_ml",
                       auto_ml_path="/configure_files/automl_params",
                       selector_config_path="/configure_files/selector_params")

core_chain.run(**entity_dict)
print("finished")
