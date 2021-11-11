# -*- coding: utf-8 -*-

# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
from pipeline.dispatch_pipeline.dispatch_udf_modeling_graph import UdfModelingGraph

def main_train(name="udf", user_configure=None, system_configure=None):
    if user_configure is None:
        user_configure = {'data_clear_flag': True, 'data_file_type': 'csv', 'dataset_name': 'plaindataset', 'dataset_weight_dict': None, 'feature_configure_path': '/home/liangqian/文档/公开数据集/bank/bank_skip.yaml', 'feature_generator_flag': True, 'init_model_root': None, 'label_switch_type': None, 'loss_name': None, 'metric_eval_used_flag': False, 'metric_name': 'binary_f1', 'mode': 'udf', 'model_zoo': ['lightgbm'], 'supervised_feature_selector_flag': False, 'supervised_selector_mode': 'topk_select', 'target_names': ['deposit'], 'task_name': 'binary_classification', 'train_column_name_flag': True, 'train_data_path': '/home/liangqian/文档/公开数据集/bank/bank.csv', 'unsupervised_feature_selector_flag': False, 'use_weight_flag': False, 'val_column_name_flag': True, 'val_data_path': None, 'weight_column_name': None, 'work_root': '/home/liangqian/Gauss/experiments/j97qq5-GBDTSelector_random_search_lightgbm'}

    if system_configure is None:
        system_configure = {'auto_ml_path': '/home/liangqian/Gauss/configure_files/automl_params', 'selector_configure_path': '/home/liangqian/Gauss/configure_files/selector_params', 'improved_selector_configure_path': '/home/liangqian/Gauss/configure_files/improved_selector_params', 'auto_ml_name': 'tabular_auto_ml', 'data_clear_name': 'plain_data_clear', 'feature_generator_name': 'featuretools_generation', 'feature_configure_name': 'feature_configure', 'label_encoder_name': 'plain_label_encoder', 'label_encoder_flag': True, 'opt_model_names': ['tpe', 'random_search', 'anneal', 'evolution'], 'supervised_feature_selector_name': 'supervised_feature_selector', 'improved_supervised_feature_selector_name': 'improved_supervised_feature_selector', 'supervised_selector_model_names': ['GBDTSelector'], 'selector_trial_num': 5, 'auto_ml_trial_num': 15, 'feature_model_trial': 15, 'max_exec_duration': 36000, 'type_inference_name': 'plain_type_inference', 'unsupervised_feature_selector_name': 'unsupervised_feature_selector', 'model_need_clear_flag': {'lightgbm': False, 'lr': True, 'dnn': True}}

    model_graph = UdfModelingGraph(name=name,
                                   user_configure=user_configure,
                                   system_configure=system_configure)

    model_graph.run()


main_train()
