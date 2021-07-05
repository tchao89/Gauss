# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

class Inference(Object):
   
    def __init__(self,
                 name: str,
                 work_root: str,
                 out_put_path: str,
                 ):  
        self.name = name
        self.work_root = work_root
        self.root_conf = self.work_root + "/" + "pipline.configure"
        self.conf = parse_conf(self.root_conf)
        self.task_type = conf.task_type
        self.metric_name = conf.metric_name
        self.test_data_path = test_data_path
        self.data_clear_flag = conf.data_clear_flag
        self.feature_generator_flag = conf.feature_generator_flag
        self.unsupervised_feature_name = conf.unsupervised_feature_name
        self.supervised_feature_selector_flag = conf.supervised_feature_selector_flag
        self.dataset_name = conf.dataset_name
        self.type_inference_name = conf.type_inference_name
        self.data_clear_name = conf.data_clear_name
        self.feature_generator_name = conf.feature_generator_name
        self.data_clear_flag = conf.data_clear_flag
        self.feature_generator_flag = conf.feature_generator_flag
        self.unsupervised_feature_selector_flag = conf.unsupervised_feature_selector_flag
        self.supervised_feature_selector_flag = conf.supervised_feature_selector_flag
    def output_result(predict_result):


    def run(self):
        work_feaure_root = self.work_root + "/feature"
        feature_dict={}
        feature_dict["user_feature"] = "null"
        feature_dict["type_inference_feature"] = work_feaure_root + "/." + type_inference_feature_path
        feature_dict["feature_generator_feature"] = work_feaure_root + "/." + feature_generate_path
        feature_dict["unsupervised_feature"] = work_feaure_root + "/." + unsupervise_feature_selector_path
        feature_dict["supervised_feature"] = work_feaure_root + "/." + supervise_feature_selector_path
        preprocess_chain = PreprocessRoute("PreprocessRoute",
                                            feature_dict,
                                            self.task_type,
                                            False,
                                            None,
                                            None,
                                            self.test_data_path,
                                            self.dataset_name,
                                            self.type_inference_name,
                                            self.data_clear_name,
                                            self.data_clear_flag,
                                            self.feature_generator_name,
                                            self.feature_generator_flag,
                                            self.unsupervised_feature_selector,
                                            unsupervised_feature_selector_flag)

        entity={}
        preprocess_chain.run(entity)
        assert(entity.has_key("test_data"))
        work_model_root = work_root + "/model/" + model + "/"
        model_save_root = work_model_root + "/model_save"
        model_config_root = work_model_root + "/model_config"
        model_conf = utils.parse_conf(model_config_root)
        core_chain = CoreRoute("core_route",
                               False,
                               model_save_root,
                               feature_dict["supervised_feature"], 
                               feature_dict["unsupervised_feature"],
                               model_conf.model_name,
                               self.metric_type,
                               self.task_type,
                               out_put_choice,
                               self.feature_selector_name,
                               self.feature_selector_flag,
                               None)
        core_chain.run(entity)
        assert(entity.has_key["predict_result_dataset"])
        output_result(entity["predict_result_dataset"])
 
    
        
            


       
       