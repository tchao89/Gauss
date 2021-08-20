# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab

from __future__ import annotations

import multiprocessing
from typing import List
from multiprocessing import Pool, shared_memory

import numpy as np

from entity.dataset.multiprocess_plain_dataset import PlaintextDataset
from pipeline.core_chain import CoreRoute
from pipeline.preprocess_chain import PreprocessRoute
from utils.base import check_data
from pipeline.mapping import EnvironmentConfigure
from pipeline.base_modeling_tree import BaseModelingTree

from utils.exception import PipeLineLogicError
from utils.Logger import logger
from utils.bunch import Bunch


# pipeline defined by user.
class MultiprocessUdfModelingTree(BaseModelingTree):
    def __init__(self, name: str, work_root: str, task_type: str, metric_name: str, train_data_path: str,
                 val_data_path: str = None, target_names=None, feature_configure_path: str = None,
                 dataset_type: str = "plain", type_inference: str = "plain", data_clear: str = "plain",
                 data_clear_flag=None, feature_generator: str = "featuretools", feature_generator_flag=None,
                 unsupervised_feature_selector: str = "unsupervised", unsupervised_feature_selector_flag=None,
                 supervised_feature_selector: str = "supervised", supervised_feature_selector_flag=None, model_zoo=None,
                 supervised_selector_names=None, auto_ml: str = "plain", opt_model_names: List[str] = None):
        """
        :param name:
        :param work_root:
        :param task_type:
        :param metric_name:
        :param train_data_path:
        :param val_data_path:
        :param feature_configure_path:
        :param dataset_type:
        :param type_inference:
        :param data_clear:
        :param data_clear_flag:
        :param feature_generator:
        :param feature_generator_flag:
        :param unsupervised_feature_selector:
        :param unsupervised_feature_selector_flag:
        :param supervised_feature_selector:
        :param supervised_feature_selector_flag:
        :param model_zoo: model name list
        :param auto_ml: auto ml name
        """

        super().__init__(name, work_root, task_type, metric_name, train_data_path, val_data_path, target_names,
                         feature_configure_path, dataset_type, type_inference, data_clear, feature_generator,
                         unsupervised_feature_selector, supervised_feature_selector, auto_ml, opt_model_names)
        if model_zoo is None:
            model_zoo = ["xgboost", "lightgbm", "catboost", "lr_lightgbm", "dnn"]

        if supervised_feature_selector_flag is None:
            supervised_feature_selector_flag = [True, False]

        if unsupervised_feature_selector_flag is None:
            unsupervised_feature_selector_flag = [True, False]

        if feature_generator_flag is None:
            feature_generator_flag = [True, False]

        if data_clear_flag is None:
            data_clear_flag = [True, False]

        self.data_clear_flag = data_clear_flag
        self.feature_generator_flag = feature_generator_flag
        self.unsupervised_feature_selector_flag = unsupervised_feature_selector_flag
        self.supervised_feature_selector_flag = supervised_feature_selector_flag
        self.model_zoo = model_zoo

        self.already_data_clear = None
        self.best_model = None
        self.best_metric = None
        self.best_result_root = None
        self.best_model_name = None

        self._model_names = model_zoo
        self._supervised_selector_names = supervised_selector_names
        self._auto_ml_names = opt_model_names
        self.jobs = None
        self.shared_entity = dict()

    def run_route(self,
                  data_clear_flag,
                  feature_generator_flag,
                  unsupervised_feature_selector_flag,
                  supervised_feature_selector_flag,
                  auto_ml_path,
                  selector_config_path):

        work_root = self.work_root + "/"

        pipeline_configure = {"data_clear_flag": data_clear_flag,
                              "feature_generator_flag": feature_generator_flag,
                              "unsupervised_feature_selector_flag": unsupervised_feature_selector_flag,
                              "supervised_feature_selector_flag": supervised_feature_selector_flag,
                              "metric_name": self.metric_name,
                              "task_type": self.task_type
                              }

        work_feature_root = work_root + "/feature"

        feature_dict = {"user_feature": self.feature_configure_path,
                        "type_inference_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().type_inference_feature,
                        "data_clear_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().data_clear_feature,
                        "feature_generator_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().feature_generator_feature,
                        "unsupervised_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().unsupervised_feature,
                        "supervised_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().supervised_feature,
                        "label_encoding_path": work_feature_root + "/" + EnvironmentConfigure.feature_dict().label_encoding_path,
                        "impute_path": work_feature_root + "/" + EnvironmentConfigure.feature_dict().impute_path,
                        "final_feature_config": work_feature_root + "/" + EnvironmentConfigure.feature_dict().final_feature_config}

        preprocess_chain = self._preprocessing_run_route(feature_dict,
                                                         data_clear_flag,
                                                         feature_generator_flag,
                                                         unsupervised_feature_selector_flag)

        entity_dict = preprocess_chain.entity_dict
        self.already_data_clear = preprocess_chain.already_data_clear

        assert "dataset" in entity_dict and "val_dataset" in entity_dict
        dataset = entity_dict["dataset"]
        val_dataset = entity_dict["val_dataset"]

        shared_memory_data = None
        shared_memory_target = None
        shared_memory_target_names = None
        shared_memory_val_data = None
        shared_memory_val_target = None
        shared_memory_val_target_names = None

        try:
            data = dataset.get_dataset().data.values
            target = dataset.get_dataset().target.values
            target_names = np.array(dataset.get_dataset().target_names)

            shared_memory_data = shared_memory.SharedMemory(create=True, size=data.nbytes)
            shared_memory_target = shared_memory.SharedMemory(create=True, size=target.nbytes)
            shared_memory_target_names = shared_memory.SharedMemory(create=True, size=target_names.nbytes)

            buffer = shared_memory_data.buf
            buffer_target = shared_memory_target.buf
            buffer_target_names = shared_memory_target_names.buf

            shared_data = np.ndarray(data.shape, dtype=data.dtype, buffer=buffer)
            shared_target = np.ndarray(target.shape, dtype=target.dtype, buffer=buffer_target)
            shared_target_names = np.ndarray(target_names.shape, dtype=target_names.dtype, buffer=buffer_target_names)

            shared_data[:] = data[:]
            shared_target[:] = target[:]
            shared_target_names[:] = target_names[:]

            val_data = val_dataset.get_dataset().data.values
            val_target = val_dataset.get_dataset().target.values
            val_target_names = target_names

            shared_memory_val_data = shared_memory.SharedMemory(create=True, size=val_data.nbytes)
            shared_memory_val_target = shared_memory.SharedMemory(create=True, size=val_target.nbytes)
            shared_memory_val_target_names = shared_memory.SharedMemory(create=True, size=val_target_names.nbytes)

            val_buffer = shared_memory_val_data.buf
            val_buffer_target = shared_memory_val_target.buf
            val_buffer_target_names = shared_memory_val_target_names.buf

            shared_val_data = np.ndarray(val_data.shape, dtype=val_data.dtype, buffer=val_buffer)
            shared_val_target = np.ndarray(val_target.shape, dtype=val_target.dtype, buffer=val_buffer_target)
            shared_val_target_names = np.ndarray(val_target_names.shape, dtype=val_target_names.dtype, buffer=val_buffer_target_names)

            shared_val_data[:] = val_data[:]
            shared_val_target[:] = val_target[:]
            shared_val_target_names[:] = val_target_names[:]

            data_pair = Bunch(data=shared_data, target=shared_target, target_names=shared_target_names)
            self.shared_entity["dataset"] = PlaintextDataset(name="train_data", task_type=self.task_type,
                                                             data_pair=data_pair)

            data_pair = Bunch(data=shared_val_data, target=shared_val_target, target_names=shared_val_target_names)
            self.shared_entity["val_dataset"] = PlaintextDataset(name="train_data", task_type=self.task_type,
                                                                 data_pair=data_pair)

            pipeline_params = []
            for supervised_selector in self._supervised_selector_names:
                for auto_ml in self._auto_ml_names:
                    for model in self._model_names:
                        # 如果未进行数据清洗, 并且模型需要数据清洗, 则返回None.
                        if check_data(already_data_clear=self.already_data_clear, model_name=model) is True:
                            pipeline_params.append(
                                [feature_dict, work_root, supervised_feature_selector_flag, auto_ml_path,
                                 selector_config_path, model, supervised_selector, auto_ml])

            self.jobs = min(len(pipeline_params), multiprocessing.cpu_count())
            assert isinstance(self.jobs, int)
            with Pool(self.jobs) as pool:
                pipeline_result = pool.map(self._single_run_route, pipeline_params)

        finally:

            shared_memory_data.unlink()
            shared_memory_target.unlink()
            shared_memory_target_names.unlink()
            shared_memory_val_data.unlink()
            shared_memory_val_target.unlink()
            shared_memory_val_target_names.unlink()

        for result in pipeline_result:
            local_metric = result[1]
            assert local_metric is not None
            local_model = result[0]
            self.update_best(local_model, local_metric, work_root, pipeline_configure)

    def _run(self):
        self.run_route(data_clear_flag=self.data_clear_flag,
                       feature_generator_flag=self.feature_generator_flag,
                       unsupervised_feature_selector_flag=self.unsupervised_feature_selector_flag,
                       supervised_feature_selector_flag=self.supervised_feature_selector_flag,
                       auto_ml_path="/home/liangqian/PycharmProjects/Gauss/configure_files/automl_config",
                       selector_config_path="/home/liangqian/PycharmProjects/Gauss/configure_files/selector_config")

    def _preprocessing_run_route(self,
                                 feature_dict,
                                 data_clear_flag,
                                 feature_generator_flag,
                                 unsupervised_feature_selector_flag):

        preprocess_chain = PreprocessRoute(name="PreprocessRoute",
                                           feature_path_dict=feature_dict,
                                           task_type=self.task_type,
                                           train_flag=True,
                                           train_data_path=self.train_data_path,
                                           val_data_path=self.val_data_path,
                                           test_data_path=None,
                                           target_names=self.target_names,
                                           type_inference_name="typeinference",
                                           data_clear_name="plaindataclear",
                                           data_clear_flag=data_clear_flag,
                                           feature_generator_name="featuretools",
                                           feature_generator_flag=feature_generator_flag,
                                           feature_selector_name="unsupervised",
                                           feature_selector_flag=unsupervised_feature_selector_flag)

        try:
            preprocess_chain.run()
        except PipeLineLogicError as e:
            logger.info(e)
            return None

        return preprocess_chain

    def _single_run_route(self, params):

        feature_dict = params[0]
        work_root = params[1]
        supervised_feature_selector_flag = params[2]
        auto_ml_path = params[3]
        selector_config_path = params[4]
        model_name = params[5]
        supervised_selector_name = params[6]
        auto_ml_name = params[7]
        entity_dict = self.shared_entity

        work_model_root = work_root + "/model/" + model_name
        model_save_root = work_model_root + "/model_save"
        model_config_root = work_model_root + "/model_parameters"
        feature_config_root = work_model_root + "/feature_config"

        core_chain = CoreRoute(name="core_route",
                               train_flag=True,
                               model_save_root=model_save_root,
                               model_config_root=model_config_root,
                               feature_config_root=feature_config_root,
                               target_feature_configure_path=feature_dict["final_feature_config"],
                               pre_feature_configure_path=feature_dict["unsupervised_feature"],
                               model_name=model_name,
                               label_encoding_path=feature_dict["label_encoding_path"],
                               model_type="tree_model",
                               metrics_name=self.metric_name,
                               task_type=self.task_type,
                               feature_selector_name="feature_selector",
                               feature_selector_flag=supervised_feature_selector_flag,
                               supervised_selector_name=supervised_selector_name,
                               auto_ml_type="auto_ml",
                               auto_ml_name=auto_ml_name,
                               auto_ml_path=auto_ml_path,
                               selector_config_path=selector_config_path)

        core_chain.run(**entity_dict)
        local_metric = core_chain.optimal_metrics
        assert local_metric is not None
        local_model = core_chain.optimal_model
        return [local_model, local_metric, auto_ml_name]
