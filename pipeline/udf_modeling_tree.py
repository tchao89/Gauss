# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

class UdfModelingTree(Object):
   
    def __init__(self,
                 name: str,
                 work_root: str,
                 task_type: str,
                 metric_name: str,
                 label_name: [],
                 train_data_path: str,
                 val_data_path: str=None,
                 feature_configue_path: str=None,
                 dataset_type: str="plain",
                 type_inference: str="plain",
                 data_clear: str="plain",
                 data_clear_flag: []=[True, False],
                 feature_generator: str="featuretools",
                 feature_generator_flag: []=[True, False],
                 unsupervised_feature_selector: str="unsupervised",
                 unsupervised_feature_selector_flag: []=[True, False],
                 supervised_feature_selector: str="supervised",
                 supervised_feature_selector_flag: []=[True, False],
                 model_zoo: list=["xgb", "lightgbm", "cgb", "lr_lightgbm", "dnn"],
                 auto_ml: str="plain"
                 ):  
        self.name = name
        self.work_root = work_root
        self.task_type = task_type
        self.metric_name = metric_name
        self.label_name = label_name
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.feature_configue_path = feature_configue_path
        self.dataset_type=dataset_type
        self.type_inference=type_inference
        self.data_clear=data_clear
        self.data_clear_flag = data_clear_flag
        self.feature_generator=feature_generator
        self.feature_generator_flag = feature_generator_flag
        self.unsupervised_feature_selector = unsupervised_feature_selector
        self.unsupervised_feature_selector_flag = unsupervised_feature_selector_flag 
        self.supervised_feature_selector=supervised_feature_selector
        self.supervised_feature_selector_flag = supervised_feature_selector_flag
        self.model_zoo=model_zoo
        self.auto_ml=auto_ml
        self.need_data_clear = False
        self.best_model="XXX"
        self.best_metric=None
        self.best_result_root="XXX"

    def run_route(folder_prefix_str,
                  data_clear_flag,
                  feature_generator_flag,
                  unsupervised_feature_generator_flag, 
                  supervised_feature_selector_flag,
                  model):
        work_root = self.work_root + "/" + folder_prefix_str
        pipline_configure_path = work_root + "/" + "pipline.configure"
        pipline_configure = "data_clear_flag" : data_clear_flag, "feature_generator_flag" : feature_generator_flag...
        utils.write(pipline_configure_path, pipline_configure)
        work_feaure_root = work_root + "/feature"
        utils.mkdir(work_root)
        utils.mkdir(work_feaure_root)
        utils.mkdir(work_model_root)
        utils.mkdir(model_save_root)
        utils.mkdir(model_config_root)
        feature_dict={}
        feature_dict["user_feature"] = self.feature_configue_path
        feature_dict["type_inference_feature"] = work_feaure_root + "/." + type_inference_feature_path
        feature_dict["feature_generator_feature"] = work_feaure_root + "/." + feature_generate_path
        feature_dict["unsupervised_feature"] = work_feaure_root + "/." + unsupervise_feature_selector_path
        feature_dict["supervised_feature"] = work_feaure_root + "/." + supervise_feature_selector_path
        preprocess_chain = PreprocessRoute("PreprocessRoute",
                                            feature_dict,
                                            self.task_type,
                                            True,
                                            self.train_data_path,
                                            self.val_data_path,
                                            None,
                                            self.dataset_name,
                                            self.type_inference_name,
                                            self.data_clear_name,
                                            data_clear_flag,
                                            self.feature_generator_name,
                                            feature_generator_flag,
                                            self.unsupervised_feature_selector,
                                            unsupervised_feature_selector_flag)

        entity={}
        preprocess_chain.run(entity)
        self.need_data_clear = preprocess_chain.need_data_clear
        check_data(self.need_data_clear, model)
        assert(entity.has_key("train_data") and entity.has_key("val_data"))
        work_model_root = work_root + "/model/" + model + "/"
        model_save_root = work_model_root + "/model_save"
        model_config_root = work_model_root + "/model_config"
        core_chain = CoreRoute("core_route",
                                   True,
                                   model_save_root,
                                   feature_dict["supervised_feature"], 
                                   feature_dict["unsupervised_feature"],
                                   model,
                                   self.metric_type,
                                   self.task_type,
                                   self.feature_selector_name,
                                   feature_selector_flag,
                                   self.auto_ml_type)
        core_chain.run(entity)
        local_metric = core_chain.get_val_metric()
        return local_metric, work_root
    def update_best(local_best_model, local_best_metric, local_best_work_root):
        if compare(local_best_metric, self.best_metric) < 0
            self.best_model = local_best_metric
            self.best_metric = local_best_metric
            self.best_result_root = local_best_work_root
    def run(self):
        for data_clear in self.data_clear_flag:
            for feature_generator in self.feature_generator_flag:
                for unsupervised_feature_sel in self.unsupervised_feature_selector_flag:
                    for supervise_feature_sel in self.supervised_feature_selector_flag:
                        for model in self.model_zoo:
                            prefix = str(data_clear) + "_" + str(feature_generator) + "_" 
                                    + str(unsupervised_feature_sel) + "_" + str(supervise_feature_sel)
                            update_best(model, run_route(prefix, data_clear, feature_generator, unsupervised_feature_sel,
                             supervise_feature_sel, [model]))
 