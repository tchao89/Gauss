# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
"""
preprocessing pipeline, used for type inference, data clear,
feature generation and unsupervised feature selector.
"""
from __future__ import annotations

import os.path

from utils.bunch import Bunch
from utils.exception import PipeLineLogicError
from utils.Logger import logger
from utils.constant_values import ConstantValues

from gauss.component import Component
from gauss_factory.gauss_factory_producer import GaussFactoryProducer


class PreprocessRoute(Component):
    """
    PreprocessRoute object.
    """

    def __init__(self, **params):
        """

        :param name: PreprocessRoute name.
        :param feature_path_dict: feature config file path
        :param task_name: classification or regression.
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
        :param unsupervised_feature_selector_name: name for gauss factory to
        create feature selector(unsupervised).
        :param unsupervised_feature_selector_flag: bool value, if true, feature
        selector will be used(unsupervised).
        """
        assert isinstance(params[ConstantValues.train_flag], bool)
        assert params[ConstantValues.task_name] in [ConstantValues.classification,
                                                    ConstantValues.regression]

        super().__init__(
            name=params["name"],
            train_flag=params["train_flag"],
            task_name=params["task_name"]
        )

        # 用户提供的特征说明文件
        assert ConstantValues.user_feature_path in params[ConstantValues.feature_path_dict]
        # 类型推导生成的特征说明文件
        assert ConstantValues.type_inference_feature_path in params[ConstantValues.feature_path_dict]
        # 数据清洗的特征说明文件
        assert ConstantValues.data_clear_feature_path in params[ConstantValues.feature_path_dict]
        # 特征生成的特征说明文件
        assert ConstantValues.feature_generator_feature_path in params[ConstantValues.feature_path_dict]
        # 无监督特征选择的特征说明文件
        assert ConstantValues.unsupervised_feature_path in params[ConstantValues.feature_path_dict]
        # label encoding file path, .db文件
        assert ConstantValues.label_encoding_models_path in params[ConstantValues.feature_path_dict]

        self._entity_dict = None

        self._feature_generator_flag = params["feature_generator_flag"]
        self._already_data_clear = None

        self._val_data_path = params["val_data_path"]
        self._train_data_path = params["train_data_path"]
        self._test_data_path = params["test_data_path"]
        self._target_names = params.get("target_names")
        self._dataset_name = params["dataset_name"]

        # Create component algorithms.
        logger.info("Creating type inference object.")
        self.type_inference = self.create_component(
            component_name=params["type_inference_name"],
            **Bunch(
                name=params["type_inference_name"],
                task_name=params["task_name"],
                train_flag=self._train_flag,
                source_file_path=params["feature_path_dict"][ConstantValues.user_feature_path],
                final_file_path=params["feature_path_dict"][ConstantValues.type_inference_feature_path],
                final_file_prefix="final"
            )
        )

        logger.info("Creating data clear object.")
        self.data_clear = self.create_component(
            component_name=params["data_clear_name"],
            **Bunch(
                name=params["data_clear_name"],
                train_flag=self._train_flag,
                enable=params["data_clear_flag"],
                task_name=params["task_name"],
                feature_configure_path=params["feature_path_dict"][ConstantValues.type_inference_feature_path],
                data_clear_configure_path=params["feature_path_dict"][ConstantValues.impute_models_path],
                final_file_path=params["feature_path_dict"][ConstantValues.data_clear_feature_path],
                strategy_dict=None
            )
        )

        logger.info("Creating label encoding object.")
        self.label_encoder = self.create_component(
            component_name=params["label_encoder_name"],
            **Bunch(
                name=params["label_encoder_name"],
                train_flag=self._train_flag,
                enable=params["label_encoder_flag"],
                task_name=params["task_name"],
                feature_config_path=params["feature_path_dict"][ConstantValues.data_clear_feature_path],
                final_file_path=params["feature_path_dict"][ConstantValues.label_encoder_feature_path],
                label_encoding_configure_path=params["feature_path_dict"][ConstantValues.label_encoding_models_path],
            )
        )

        logger.info("Creating feature generator object")
        self.feature_generator = self.create_component(
            component_name=params["feature_generator_name"],
            **Bunch(
                name=params["feature_generator_name"],
                train_flag=self._train_flag,
                enable=params["feature_generator_flag"],
                task_name=params["task_name"],
                feature_config_path=params["feature_path_dict"][ConstantValues.label_encoder_feature_path],
                final_file_path=params["feature_path_dict"][ConstantValues.feature_generator_feature_path]
            )
        )

        logger.info("Creating unsupervised feature selector object.")
        self.unsupervised_feature_selector = self.create_component(
            component_name=params["unsupervised_feature_selector_name"],
            **Bunch(
                name=params["unsupervised_feature_selector_name"],
                train_flag=self._train_flag,
                enable=params["unsupervised_feature_selector_flag"],
                task_name=params["task_name"],
                feature_config_path=params["feature_path_dict"][ConstantValues.feature_generator_feature_path],
                final_file_path=params["feature_path_dict"][ConstantValues.unsupervised_feature_path]
            )
        )

    @classmethod
    def create_component(cls, component_name: str, **params):
        """
        component factory.
        :param component_name:
        :param params:
        :return:
        """
        gauss_factory = GaussFactoryProducer()
        component_factory = gauss_factory.get_factory(choice="component")
        return component_factory.get_component(
            component_name=component_name,
            **params
        )

    @classmethod
    def create_entity(cls, entity_name: str, **params):
        """
        entity factory.
        :param entity_name:
        :param params:
        :return:
        """
        gauss_factory = GaussFactoryProducer()
        entity_factory = gauss_factory.get_factory(choice="entity")
        return entity_factory.get_entity(
            entity_name=entity_name,
            **params
        )

    def _train_run(self, **entity):
        entity_dict = {}

        assert self._train_data_path is not None
        assert os.path.isfile(self._train_data_path)
        assert self._train_flag is True

        # 拼接数据
        dataset_params = Bunch(
            name=self._dataset_name,
            task_name=self._task_name,
            data_pair=None,
            data_path=self._train_data_path,
            target_name=self._target_names,
            memory_only=True
        )
        logger.info("Starting loading data...")
        train_dataset = self.create_entity(
            entity_name=self._dataset_name,
            **dataset_params
        )
        if self._val_data_path is not None:
            val_dataset_params = Bunch(
                name=self._dataset_name,
                task_name=self._task_name,
                data_pair=None,
                data_path=self._val_data_path,
                target_name=self._target_names,
                memory_only=True
            )
            val_dataset = self.create_entity(
                self._dataset_name,
                **val_dataset_params
            )
            train_dataset.union(val_dataset)

        entity_dict["train_dataset"] = train_dataset
        # 类型推导
        logger.info("Starting type inference.")
        self.type_inference.run(**entity_dict)

        # 数据清洗
        logger.info("Starting data clear.")
        self.data_clear.run(**entity_dict)

        self._already_data_clear = self.data_clear.already_data_clear
        if self._already_data_clear is False \
                and train_dataset.need_data_clear is True \
                and self._feature_generator_flag is True:
            raise PipeLineLogicError("Aberrant dataset can not generate additional features.")

        logger.info("Starting encoding features and labels.")
        self.label_encoder.run(**entity_dict)

        logger.info("Starting feature generation.")
        # 特征生成
        self.feature_generator.run(**entity_dict)

        logger.info("Starting unsupervised feature selector.")
        # 无监督特征选择
        self.unsupervised_feature_selector.run(**entity_dict)
        # 数据拆分
        val_dataset = train_dataset.split()
        entity_dict["val_dataset"] = val_dataset
        self._entity_dict = entity_dict
        logger.info("Dataset preprocessing has finished.")

    def _predict_run(self, **entity):
        entity_dict = {}

        assert self._test_data_path is not None
        assert os.path.isfile(self._test_data_path)
        assert self._train_flag is False

        dataset_params = Bunch(
            name="test",
            task_name=self._task_name,
            data_pair=None,
            data_path=self._test_data_path,
            target_name=self._target_names,
            memory_only=True
        )
        test_dataset = self.create_entity(
            entity_name=self._dataset_name,
            **dataset_params
        )
        entity_dict["infer_dataset"] = test_dataset

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
        """
        :return: bool value
        """
        return self._already_data_clear

    @property
    def entity_dict(self):
        """

        :return: dict
        """
        return self._entity_dict
