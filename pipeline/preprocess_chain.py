# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: luoqing

from __future__ import annotations

import abc

from gauss.component import Component

class PreprocessRoute(Component):
    def __init__(self,
                 name: str,
                 feature_path_dict: dict,
                 task_type: str,
                 train_flag: bool,
                 train_data_path: str,
                 val_data_path: str,
                 test_data_path: str,
                 dataset_name: str="plain",
                 type_inference_name: str="plain",
                 data_clear_name: str="plain",
                 data_clear_flag: bool=True,
                 feature_generator_name: str="featuretools",
                 feature_generator_flag: bool=True,
                 feature_selector_name: str="unsupervised",
                 feature_selector_flag: bool=True
                 ):
        """

        :param name: PreprocessRoute name
        :param feature_path_dict: feature config file path
        :param task_type: classification or regression.
        :param train_flag: bool
        :param train_data_path:
        :param val_data_path:
        :param test_data_path:
        :param dataset_name:
        :param type_inference_name:
        :param data_clear_name:
        :param data_clear_flag:
        :param feature_generator_name:
        :param feature_generator_flag:
        :param feature_selector_name:
        :param feature_selector_flag:
        """
        self.data_clear_flag = data_clear_flag
        self.feature_generator_flag = feature_generator_flag
        self.feature_selector_flag = feature_selector_flag
        self.need_data_clear = False
        type_inference = TypeInferenceFactory(type_inference_name, feature_path_dict["user_feature"], feature_path_dict["type_inference_feature"],...)
        data_clear = DataClearFactory(data_clear_name, feature_path_dict["type_inference_feature"], data_clear_flag,...)
        feature_generator = FeatureGenerator(feature_generator_name, feature_path_dict["type_inference_feature"], feature_path_dict["feature_generator_feature"], feature_generator_flag...)
        unsupervised_feature_selector= FeatureSelectorFactory(feature_selector_name, feature_path_dict["feature_generator_feature"], feature_path_dict["unsupervised_feature"], feature_selector_flag...)
        super(PreprocessRoute, self).__init__(
            name=name,
            train_flag=train_flag
        )
    def run(self, **entity):
        if self._train_flag:
            self._train_run(**entity)
        else:
            self._predict_run(**entity)
    def _train_run(self, **entity):
        entity_dict = {}
        train_dataset = DateSetFactory("Plain", train_data_path,...)
        if(val_data_path != None):
            val_dataset = DateSetFactory("Plain", val_data_path,...)
            train_dataset.union(val_dataset)
        entity_dict["train_dataset"] = train_dataset
        type_inference.run(entity_dict)
        data_clear.run(entity_dict)
        feature_generator.run(entity_dict)
        unsupervised_feature_selector.run(entity_dict)
        if(val_data_path != None):
            val_dataset = train_dataset.split()
            entity_dict["val_dataset"] = val_dataset
    def _predict_run(self, **entity):
        
