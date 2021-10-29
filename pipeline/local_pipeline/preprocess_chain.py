"""-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
preprocessing pipeline, used for type inference, data clear,
feature generation and unsupervised feature selector."""
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
        assert isinstance(params[ConstantValues.train_flag], str)
        assert params[ConstantValues.task_name] in [ConstantValues.binary_classification,
                                                    ConstantValues.multiclass_classification,
                                                    ConstantValues.regression]

        super().__init__(
            name=params[ConstantValues.name],
            train_flag=params[ConstantValues.train_flag],
            task_name=params[ConstantValues.task_name]
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

        self._feature_generator_flag = params[ConstantValues.feature_generator_flag]
        self._already_data_clear = None

        self._data_file_type = params[ConstantValues.data_file_type]
        self._dataset_name = params[ConstantValues.dataset_name]
        self._dataset_weight_dict = params[ConstantValues.dataset_weight_dict]

        if self._task_name == ConstantValues.regression:
            label_encoder_params = Bunch(
                name=params[ConstantValues.label_encoder_name],
                train_flag=self._train_flag,
                enable=params[ConstantValues.label_encoder_flag],
                task_name=params[ConstantValues.task_name],
                label_switch_type=params[ConstantValues.label_switch_type],
                feature_configure_path=params[ConstantValues.feature_path_dict][ConstantValues.data_clear_feature_path],
                final_file_path=params[ConstantValues.feature_path_dict][ConstantValues.label_encoder_feature_path],
                label_encoding_configure_path=params[ConstantValues.feature_path_dict][
                    ConstantValues.label_encoding_models_path]
            )
        else:
            label_encoder_params = Bunch(
                name=params[ConstantValues.label_encoder_name],
                train_flag=self._train_flag,
                enable=params[ConstantValues.label_encoder_flag],
                task_name=params[ConstantValues.task_name],
                feature_configure_path=params[ConstantValues.feature_path_dict][ConstantValues.data_clear_feature_path],
                final_file_path=params[ConstantValues.feature_path_dict][ConstantValues.label_encoder_feature_path],
                label_encoding_configure_path=params[ConstantValues.feature_path_dict][
                    ConstantValues.label_encoding_models_path]
            )
        if self._train_flag == ConstantValues.train:
            self._target_names = params[ConstantValues.target_names]
            self._use_weight_flag = params[ConstantValues.use_weight_flag]
            self._weight_column_name = params[ConstantValues.weight_column_name]
            self._train_column_name_flag = params[ConstantValues.train_column_name_flag]
            self._val_column_name_flag = params[ConstantValues.val_column_name_flag]
            self._val_data_path = params[ConstantValues.val_data_path]
            self._train_data_path = params[ConstantValues.train_data_path]

        elif self._train_flag == ConstantValues.increment:
            self._increment_column_name_flag = params[ConstantValues.increment_column_name_flag]
            self._target_names = params[ConstantValues.target_names]
            self._train_data_path = params[ConstantValues.train_data_path]

        elif self._train_flag == ConstantValues.inference:
            self._inference_column_name_flag = params[ConstantValues.inference_column_name_flag]
            assert isinstance(self._inference_column_name_flag, bool)
            self._inference_data_path = params[ConstantValues.inference_data_path]

        else:
            raise ValueError("Value: train_flag should be train, "
                             "increment or inference, but get {}".format(self._train_flag))

        # Create component algorithms.
        logger.info("Creating type inference object.")
        self.type_inference = self.create_component(
            component_name=params[ConstantValues.type_inference_name],
            **Bunch(
                name=params[ConstantValues.type_inference_name],
                task_name=params[ConstantValues.task_name],
                train_flag=self._train_flag,
                source_file_path=params[ConstantValues.feature_path_dict][ConstantValues.user_feature_path],
                final_file_path=params[ConstantValues.feature_path_dict][ConstantValues.type_inference_feature_path]
            )
        )

        logger.info("Creating data clear object.")
        self.data_clear = self.create_component(
            component_name=params[ConstantValues.data_clear_name],
            **Bunch(
                name=params[ConstantValues.data_clear_name],
                train_flag=self._train_flag,
                enable=params[ConstantValues.data_clear_flag],
                task_name=params[ConstantValues.task_name],
                feature_configure_path=params[ConstantValues.feature_path_dict][
                    ConstantValues.type_inference_feature_path],
                data_clear_configure_path=params[ConstantValues.feature_path_dict][ConstantValues.impute_models_path],
                final_file_path=params[ConstantValues.feature_path_dict][ConstantValues.data_clear_feature_path],
                strategy_dict=None
            )
        )

        logger.info("Creating label encoding object.")
        self.label_encoder = self.create_component(
            component_name=params[ConstantValues.label_encoder_name],
            **label_encoder_params
        )

        logger.info("Creating feature generator object")
        self.feature_generator = self.create_component(
            component_name=params[ConstantValues.feature_generator_name],
            **Bunch(
                name=params[ConstantValues.feature_generator_name],
                train_flag=self._train_flag,
                enable=params[ConstantValues.feature_generator_flag],
                task_name=params[ConstantValues.task_name],
                feature_config_path=params[ConstantValues.feature_path_dict][ConstantValues.label_encoder_feature_path],
                final_file_path=params[ConstantValues.feature_path_dict][ConstantValues.feature_generator_feature_path]
            )
        )

        logger.info("Creating unsupervised feature selector object.")
        self.unsupervised_feature_selector = self.create_component(
            component_name=params[ConstantValues.unsupervised_feature_selector_name],
            **Bunch(
                name=params[ConstantValues.unsupervised_feature_selector_name],
                train_flag=self._train_flag,
                enable=params[ConstantValues.unsupervised_feature_selector_flag],
                task_name=params[ConstantValues.task_name],
                feature_config_path=params[ConstantValues.feature_path_dict][
                    ConstantValues.feature_generator_feature_path],
                final_file_path=params[ConstantValues.feature_path_dict][ConstantValues.unsupervised_feature_path]
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
        assert self._train_flag is ConstantValues.train

        # 拼接数据
        dataset_params = Bunch(
            name=ConstantValues.train_dataset,
            task_name=self._task_name,
            data_package=None,
            data_path=self._train_data_path,
            data_file_type=self._data_file_type,
            target_names=self._target_names,
            use_weight_flag=self._use_weight_flag,
            dataset_weight_dict=self._dataset_weight_dict,
            weight_column_name=self._weight_column_name,
            column_name_flag=self._train_column_name_flag,
            memory_only=True
        )
        logger.info("Starting loading data.")
        train_dataset = self.create_entity(
            entity_name=self._dataset_name,
            **dataset_params
        )
        if self._val_data_path is not None:
            val_dataset_params = Bunch(
                name=ConstantValues.val_dataset,
                task_name=self._task_name,
                data_package=None,
                data_path=self._val_data_path,
                data_file_type=self._data_file_type,
                target_names=self._target_names,
                use_weight_flag=self._use_weight_flag,
                dataset_weight_dict=self._dataset_weight_dict,
                weight_column_name=self._weight_column_name,
                column_name_flag=self._val_column_name_flag,
                memory_only=True
            )
            val_dataset = self.create_entity(
                self._dataset_name,
                **val_dataset_params
            )
            train_dataset.union(val_dataset)

        entity_dict[ConstantValues.train_dataset] = train_dataset
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
        entity_dict[ConstantValues.val_dataset] = val_dataset
        logger.info("Dataset preprocessing has finished.")
        return entity_dict

    def _increment_run(self, **entity):
        entity_dict = {}

        assert self._train_data_path is not None
        assert os.path.isfile(self._train_data_path)
        assert self._train_flag is ConstantValues.increment

        # 拼接数据
        dataset_params = Bunch(
            name=ConstantValues.increment_dataset,
            task_name=self._task_name,
            data_package=None,
            data_path=self._train_data_path,
            data_file_type=self._data_file_type,
            column_name_flag=self._increment_column_name_flag,
            target_names=self._target_names,
            memory_only=True
        )
        logger.info("Starting loading data.")
        increment_dataset = self.create_entity(
            entity_name=self._dataset_name,
            **dataset_params
        )

        entity_dict[ConstantValues.increment_dataset] = increment_dataset
        # 类型推导
        logger.info("Starting type inference.")
        self.type_inference.run(**entity_dict)

        # 数据清洗
        logger.info("Starting data clear.")
        self.data_clear.run(**entity_dict)

        self._already_data_clear = self.data_clear.already_data_clear
        if self._already_data_clear is False \
                and increment_dataset.need_data_clear is True \
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
        logger.info("Dataset preprocessing has finished.")
        return entity_dict

    def _predict_run(self, **entity):
        entity_dict = {}

        assert self._inference_data_path is not None
        assert os.path.isfile(self._inference_data_path)
        assert self._train_flag is ConstantValues.inference

        dataset_params = Bunch(
            name=ConstantValues.increment_dataset,
            task_name=self._task_name,
            data_pair=None,
            data_path=self._inference_data_path,
            data_file_type=self._data_file_type,
            column_name_flag=self._inference_column_name_flag,
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

        self.label_encoder.run(**entity_dict)
        # 特征生成
        self.feature_generator.run(**entity_dict)
        # 无监督特征选择
        self.unsupervised_feature_selector.run(**entity_dict)
        return entity_dict

    @property
    def already_data_clear(self):
        """
        :return: bool value
        """
        return self._already_data_clear

    def run(self, **entity):
        """
        Run component.
        :param entity:
        :return:
        """
        if self._train_flag == ConstantValues.train:
            return self._train_run(**entity)
        elif self._train_flag == ConstantValues.inference:
            return self._predict_run(**entity)
        elif self._train_flag == ConstantValues.increment:
            return self._increment_run(**entity)
        else:
            raise ValueError(
                "Value: train_flag is illegal, and it can be in "
                "[train, inference, increment], but get {} instead.".format(
                    self._train_flag)
            )
