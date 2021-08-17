# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

import os.path
from typing import List

from utils.bunch import Bunch
from utils.exception import PipeLineLogicError
from utils.Logger import logger
from gauss.component import Component
from gauss_factory.gauss_factory_producer import GaussFactoryProducer


class PreprocessRoute(Component):
    def __init__(self,
                 name: str,
                 feature_path_dict: dict,
                 task_type: str,
                 train_flag: bool,
                 train_data_path: str = None,
                 val_data_path: str = None,
                 test_data_path: str = None,
                 target_names: List[str] = None,
                 dataset_name: str = "plaindataset",
                 type_inference_name: str = "typeinference",
                 data_clear_name: str = "plaindataclear",
                 data_clear_flag: bool = True,
                 feature_generator_name: str = "featuretools",
                 feature_generator_flag: bool = True,
                 feature_selector_name: str = "unsupervised",
                 feature_selector_flag: bool = True
                 ):
        """

        :param name: PreprocessRoute name.
        :param feature_path_dict: feature config file path, including
        :param task_type: classification or regression.
        :param train_flag: bool value, if true, this object will be used to train model.
        :param train_data_path: training data path.
        :param val_data_path: validation data path.
        :param test_data_path: test data path for prediction.
        :param dataset_name: name for gauss factory to create dataset.
        :param type_inference_name: name for gauss factory to create type_inference.
        :param data_clear_name: name for gauss factory to create data_clear.
        :param data_clear_flag: bool value, if true, data_clear must execute.
        :param feature_generator_name: name for gauss factory to create feature_generator.
        :param feature_generator_flag: bool value, if true, feature generator will be used.
        :param feature_selector_name: name for gauss factory to create feature selector(unsupervised).
        :param feature_selector_flag: bool value, if true, feature selector will be used(unsupervised).
        """
        assert isinstance(train_flag, bool)
        assert task_type in ["classification", "regression"]

        super(PreprocessRoute, self).__init__(
            name=name,
            train_flag=train_flag
        )

        self._task_type = task_type

        # 用户提供的特征说明文件
        assert "user_feature" in feature_path_dict
        # 类型推导生成的特征说明文件
        assert "type_inference_feature" in feature_path_dict
        # 数据清洗的特征说明文件
        assert "data_clear_feature" in feature_path_dict
        # 特征生成的特征说明文件
        assert "feature_generator_feature" in feature_path_dict
        # 无监督特征选择的特征说明文件
        assert "unsupervised_feature"
        # label encoding file path, .db文件
        assert "label_encoding_path" in feature_path_dict

        self._entity_dict = None

        self._data_clear_flag = data_clear_flag
        self._feature_generator_flag = feature_generator_flag
        self._feature_selector_flag = feature_selector_flag
        self._already_data_clear = None

        self._val_data_path = val_data_path
        self._train_data_path = train_data_path
        self._test_data_path = test_data_path
        self._target_names = target_names

        self._dataset_name = dataset_name
        self._data_clear_name = data_clear_name
        self._data_clear_flag = data_clear_flag
        self._feature_generation_name = feature_generator_name
        self._feature_selector_name = feature_selector_name

        # Create component algorithms.
        inference_params = Bunch(name=type_inference_name, task_name=task_type,
                                 train_flag=self._train_flag,
                                 source_file_path=feature_path_dict["user_feature"],
                                 final_file_path=feature_path_dict["type_inference_feature"],
                                 final_file_prefix="final")
        self.type_inference = self.create_component(component_name=type_inference_name, **inference_params)

        clear_params = Bunch(name=data_clear_name, train_flag=self._train_flag, enable=self._data_clear_flag,
                             model_name="tree_model",
                             feature_configure_path=feature_path_dict["type_inference_feature"],
                             data_clear_configure_path=feature_path_dict["impute_path"],
                             final_file_path=feature_path_dict["data_clear_feature"], strategy_dict=None)
        self.data_clear = self.create_component(component_name="plaindataclear", **clear_params)

        generation_params = Bunch(name=feature_generator_name, train_flag=self._train_flag,
                                  enable=self._feature_generator_flag,
                                  feature_config_path=feature_path_dict["data_clear_feature"],
                                  label_encoding_configure_path=feature_path_dict["label_encoding_path"],
                                  final_file_path=feature_path_dict["feature_generator_feature"])
        self.feature_generator = self.create_component(component_name="featuretoolsgeneration", **generation_params)

        u_params = Bunch(name=feature_selector_name, train_flag=self._train_flag, enable=self._feature_selector_flag,
                         label_encoding_configure_path=feature_path_dict["label_encoding_path"],
                         feature_config_path=feature_path_dict["feature_generator_feature"],
                         final_file_path=feature_path_dict["unsupervised_feature"])
        self.unsupervised_feature_selector = self.create_component(component_name="unsupervisedfeatureselector",
                                                                   **u_params)

    @classmethod
    def create_component(cls, component_name: str, **params):
        gauss_factory = GaussFactoryProducer()
        component_factory = gauss_factory.get_factory(choice="component")
        return component_factory.get_component(component_name=component_name, **params)

    @classmethod
    def create_entity(cls, entity_name: str, **params):
        gauss_factory = GaussFactoryProducer()
        entity_factory = gauss_factory.get_factory(choice="entity")
        return entity_factory.get_entity(entity_name=entity_name, **params)

    def run(self, **entity):
        if self._train_flag:
            return self._train_run(**entity)
        else:
            return self._predict_run(**entity)

    def _train_run(self, **entity):
        entity_dict = {}

        assert self._train_data_path is not None
        assert os.path.isfile(self._train_data_path)
        assert self._train_flag is True
        # 拼接数据
        dataset_params = Bunch(name="test", task_type=self._task_type, data_pair=None,
                               data_path=self._train_data_path,
                               target_name=self._target_names, memory_only=True)
        logger.info("starting loading data...")
        train_dataset = self.create_entity(entity_name="plaindataset", **dataset_params)
        if self._val_data_path is not None:
            val_dataset_params = Bunch(name="test", task_type=self._task_type, data_pair=None,
                                       data_path=self._val_data_path,
                                       target_name=self._target_names, memory_only=True)
            val_dataset = self.create_entity("plaindataset", **val_dataset_params)
            train_dataset.union(val_dataset)

        entity_dict["dataset"] = train_dataset
        # 类型推导
        logger.info("starting type inference...")
        self.type_inference.run(**entity_dict)
        
        # 数据清洗
        logger.info("starting data clear...")
        self.data_clear.run(**entity_dict)

        self._already_data_clear = self.data_clear.already_data_clear
        if self._already_data_clear is False and train_dataset.need_data_clear is True and self._feature_generator_flag is True:
            raise PipeLineLogicError("Aberrant dataset can not generate additional features.")

        logger.info("starting feature generation...")
        # 特征生成
        self.feature_generator.run(**entity_dict)

        logger.info("starting unsupervised feature selector")
        # 无监督特征选择
        self.unsupervised_feature_selector.run(**entity_dict)
        # 数据拆分
        val_dataset = train_dataset.split()
        entity_dict["val_dataset"] = val_dataset
        self._entity_dict = entity_dict

    def _predict_run(self, **entity):
        entity_dict = {}

        assert self._test_data_path is not None
        assert os.path.isfile(self._test_data_path)
        assert self._train_flag is False

        dataset_params = Bunch(name="test", task_type=self._task_type, data_pair=None,
                               data_path=self._test_data_path,
                               target_name=self._target_names, memory_only=True)
        test_dataset = self.create_entity(entity_name="plaindataset", **dataset_params)
        entity_dict["dataset"] = test_dataset

        self.type_inference.run(**entity_dict)
        # 数据清洗
        self.data_clear.run(**entity_dict)
        # 特征生成
        self.feature_generator.run(**entity_dict)
        # 无监督特征选择
        self.unsupervised_feature_selector.run(**entity_dict)

        self._entity_dict = entity_dict

    @property
    def already_data_clear(self):
        return self._already_data_clear

    @property
    def entity_dict(self):
        return self._entity_dict
