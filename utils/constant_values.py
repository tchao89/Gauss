"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""

class ConstantValues:
    # system name
    name = "name"
    train_flag = "train_flag"
    increment_flag = "increment_flag"
    enable = "enable"
    task_name = "task_name"
    target_names = "target_names"
    memory_only = "memory_only"
    decay_rate = "decay_rate"
    infer_result_type = "infer_result_type"
    proportion = "proportion"
    data_file_type = "data_file_type"
    data_package = "data_package"
    label_switch_type = "label_switch_type"
    supervised_selector_model_names = "supervised_selector_model_names"
    supervised_selector_mode = "supervised_selector_mode"
    selector_trial_num = "selector_trial_num"
    auto_ml_trial_num = "auto_ml_trial_num"
    feature_model_trial = "feature_model_trial"
    opt_model_names = "opt_model_names"
    optimize_mode = "optimize_mode"
    minimize = "minimize"
    maximize = "maximize"
    model_zoo = "model_zoo"
    feature_path_dict = "feature_path_dict"
    model_need_clear_flag = "model_need_clear_flag"
    metric_eval_used_flag = "metric_eval_used_flag"
    dataset_weight = "dataset_weight"
    dataset_weight_dict = "dataset_weight_dict"
    use_weight_flag = "use_weight_flag"
    train_column_name_flag = "train_column_name_flag"
    val_column_name_flag = "val_column_name_flag"
    inference_column_name_flag = "inference_column_name_flag"
    increment_column_name_flag = "increment_column_name_flag"
    column_name_flag = "column_name_flag"
    weight_column_flag = "weight_column_flag"
    weight_column_name = "weight_column_name"
    dataset_items = ["data",
                     "target",
                     "feature_names",
                     "target_names",
                     "generated_feature_names",
                     "dataset_weight",
                     "proportion",
                     "categorical_list",
                     "label_class"]
    switch_types = ["log",
                    "exp",
                    "pow",
                    None]
    # train_flag:
    train = "train"
    increment = "increment"
    inference = "inference"
    # task_name
    binary_classification = "binary_classification"
    multiclass_classification = "multiclass_classification"
    regression = "regression"
    # selector name
    GBDTSelector = "GBDTSelector"
    gradient_feature_selector = "gradient_feature_selector"
    # entity name
    metric = "metric"
    loss = "loss"
    model_name = "model_name"
    selector_model = "selector_model"
    selector_metric = "selector_metric"
    dataset_name = "dataset_name"
    train_dataset = "train_dataset"
    val_dataset = "val_dataset"
    infer_dataset = "infer_dataset"
    increment_dataset = "increment_dataset"
    metric_name = "metric_name"
    loss_name = "loss_name"
    metric_result = "metric_result"
    feature_configure_name = "feature_configure_name"
    # component name
    auto_ml = "auto_ml"
    selector_auto_ml = "selector_auto_ml"
    data_clear_name = "data_clear_name"
    type_inference_name = "type_inference_name"
    label_encoder_name = "label_encoder_name"
    feature_generator_name = "feature_generator_name"
    unsupervised_feature_selector_name = "unsupervised_feature_selector_name"
    supervised_feature_selector_name = "supervised_feature_selector_name"
    improved_supervised_feature_selector_name = "improved_supervised_feature_selector_name"
    auto_ml_name = "auto_ml_name"
    feature_selector_model_names = "feature_selector_model_names"
    # flag name
    data_clear_flag = "data_clear_flag"
    label_encoder_flag = "label_encoder_flag"
    feature_generator_flag = "feature_generator_flag"
    unsupervised_feature_selector_flag = "unsupervised_feature_selector_flag"
    supervised_feature_selector_flag = "supervised_feature_selector_flag"
    # path name
    work_root = "work_root"
    data_path = "data_path"
    init_model_root = "init_model_root"
    work_model_root = "work_model_root"
    auto_ml_path = "auto_ml_path"
    improved_selector_configure_path = "improved_selector_configure_path"
    model_root_path = "model_root_path"
    selector_configure_path = "selector_configure_path"
    feature_configure_path = "feature_configure_path"
    source_file_path = "source_file_path"
    final_file_path = "final_file_path"
    train_data_path = "train_data_path"
    val_data_path = "val_data_path"
    inference_data_path = "inference_data_path"
    user_feature_path = "user_feature_path"
    type_inference_feature_path = "type_inference_feature_path"
    data_clear_feature_path = "data_clear_feature_path"
    feature_generator_feature_path = "feature_generator_feature_path"
    unsupervised_feature_path = "unsupervised_feature_path"
    supervised_feature_path = "supervised_feature_path"
    label_encoding_models_path = "label_encoding_models_path"
    impute_models_path = "impute_models_path"
    label_encoder_feature_path = "label_encoder_feature_path"
    pre_feature_configure_path = "pre_feature_configure_path"
    target_feature_configure_path = "target_feature_configure_path"
    out_put_path = "out_put_path"
    # pipeline name
    PreprocessRoute = "PreprocessRoute"
    CoreRoute = "CoreRoute"
    # folder_name
    model = "model"
    feature = "feature"
    feature_configure = "feature_configure"
    model_parameters = "model_parameters"
    model_save = "model_save"
    # configure dict name
    final_feature_configure = "final_feature_configure"
