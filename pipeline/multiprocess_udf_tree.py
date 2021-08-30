# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
from __future__ import annotations

import gc
from typing import List
from multiprocessing import Pool, shared_memory, cpu_count

import numpy as np

from entity.dataset.multiprocess_plain_dataset import MultiprocessPlaintextDataset
from pipeline.core_chain import CoreRoute
from pipeline.preprocess_chain import PreprocessRoute
from pipeline.mapping import EnvironmentConfigure
from pipeline.base_modeling_tree import BaseModelingTree
from utils.common_component import yaml_write

from utils.exception import PipeLineLogicError, NoResultReturnException
from utils.base import get_current_memory_gb
from utils.check_dataset import check_data
from utils.Logger import logger
from utils.bunch import Bunch
from utils.callback import multiprocess_callback


# pipeline defined by user.
class MultiprocessUdfModelingTree(BaseModelingTree):
    def __init__(self, name: str, work_root: str, task_type: str, metric_name: str, train_data_path: str,
                 val_data_path: str = None, target_names=None, feature_configure_path: str = None,
                 dataset_type: str = "plain", type_inference: str = "plain", data_clear: str = "plain",
                 data_clear_flag=None, feature_generator: str = "featuretools", feature_generator_flag=None,
                 unsupervised_feature_selector: str = "unsupervised", unsupervised_feature_selector_flag=None,
                 supervised_feature_selector: str = "supervised", supervised_feature_selector_flag=None, model_zoo=None,
                 supervised_selector_names=None, auto_ml: str = "plain", opt_model_names: List[str] = None,
                 auto_ml_path: str = None, selector_config_path: str = None):
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
        super().__init__(name=name, work_root=work_root, task_type=task_type, metric_name=metric_name,
                         train_data_path=train_data_path, val_data_path=val_data_path, target_names=target_names,
                         feature_configure_path=feature_configure_path, dataset_type=dataset_type,
                         type_inference=type_inference, data_clear=data_clear, feature_generator=feature_generator,
                         unsupervised_feature_selector=unsupervised_feature_selector,
                         supervised_feature_selector=supervised_feature_selector, auto_ml=auto_ml,
                         opt_model_names=opt_model_names, auto_ml_path=auto_ml_path,
                         selector_config_path=selector_config_path)

        if model_zoo is None:
            model_zoo = ["xgboost", "lightgbm", "catboost", "lr_lightgbm", "dnn"]

        if supervised_feature_selector_flag is None:
            supervised_feature_selector_flag = True

        if unsupervised_feature_selector_flag is None:
            unsupervised_feature_selector_flag = True

        if feature_generator_flag is None:
            feature_generator_flag = True

        if data_clear_flag is None:
            data_clear_flag = True

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

        self.multiprocess_config = dict()

    def run_route(self,
                  data_clear_flag,
                  feature_generator_flag,
                  unsupervised_feature_selector_flag,
                  supervised_feature_selector_flag,
                  auto_ml_path,
                  selector_config_path):

        assert isinstance(data_clear_flag, bool) and \
               isinstance(feature_generator_flag, bool) and \
               isinstance(unsupervised_feature_selector_flag, bool) and \
               isinstance(supervised_feature_selector_flag, bool)

        work_root = self.work_root
        self.pipeline_config = {"work_root": self.work_root,
                                "data_clear_flag": data_clear_flag,
                                "data_clear": self.data_clear,
                                "feature_generator_flag": feature_generator_flag,
                                "feature_generator": self.feature_generator,
                                "unsupervised_feature_selector_flag": unsupervised_feature_selector_flag,
                                "unsupervised_feature_selector": self.unsupervised_feature_selector,
                                "supervised_feature_selector_flag": supervised_feature_selector_flag,
                                "supervised_feature_selector":self.supervised_feature_selector,
                                "metric_name": self.metric_name,
                                "task_type": self.task_type,
                                "target_names": self.target_names,
                                "dataset_type": self.dataset_type,
                                "type_inference": self.type_inference,

                                }

        work_feature_root = work_root + "/feature"

        feature_dict = {"user_feature": self.feature_configure_path,
                        "type_inference_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().type_inference_feature,
                        "data_clear_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().data_clear_feature,
                        "feature_generator_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().feature_generator_feature,
                        "unsupervised_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().unsupervised_feature,
                        "supervised_feature": work_feature_root + "/" + EnvironmentConfigure.feature_dict().supervised_feature,
                        "label_encoding_path": work_feature_root + "/" + EnvironmentConfigure.feature_dict().label_encoding_path,
                        "impute_path": work_feature_root + "/" + EnvironmentConfigure.feature_dict().impute_path}

        preprocess_chain = self._preprocessing_run_route(feature_dict,
                                                         data_clear_flag,
                                                         feature_generator_flag,
                                                         unsupervised_feature_selector_flag)

        entity_dict = preprocess_chain.entity_dict
        self.already_data_clear = preprocess_chain.already_data_clear

        assert "dataset" in entity_dict and "val_dataset" in entity_dict
        dataset = entity_dict["dataset"]
        val_dataset = entity_dict["val_dataset"]

        logger.info("Loading shared memory for dataset, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])

        shared_memory_data = None
        shared_memory_target = None
        shared_memory_target_names = None
        shared_memory_val_data = None
        shared_memory_val_target = None
        shared_memory_val_target_names = None

        try:
            logger.info("Read dataset, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            data = dataset.get_dataset().data.values
            target = dataset.get_dataset().target.values
            target_names = np.array(dataset.get_dataset().target_names)

            val_data = val_dataset.get_dataset().data.values
            val_target = val_dataset.get_dataset().target.values
            val_target_names = target_names

            logger.info("Generate shared memory for each subprocess, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            shared_memory_data = shared_memory.SharedMemory(create=True, size=data.nbytes)
            shared_memory_target = shared_memory.SharedMemory(create=True, size=target.nbytes)
            shared_memory_target_names = shared_memory.SharedMemory(create=True, size=target_names.nbytes)

            shared_memory_val_data = shared_memory.SharedMemory(create=True, size=val_data.nbytes)
            shared_memory_val_target = shared_memory.SharedMemory(create=True, size=val_target.nbytes)
            shared_memory_val_target_names = shared_memory.SharedMemory(create=True, size=val_target_names.nbytes)

            logger.info("Generate buffer for shared memory, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            buffer = shared_memory_data.buf
            buffer_target = shared_memory_target.buf
            buffer_target_names = shared_memory_target_names.buf

            val_buffer = shared_memory_val_data.buf
            val_buffer_target = shared_memory_val_target.buf
            val_buffer_target_names = shared_memory_val_target_names.buf

            logger.info("Generate shared data buffer for each subprocess, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            shared_data = np.ndarray(data.shape, dtype=data.dtype, buffer=buffer)
            shared_target = np.ndarray(target.shape, dtype=target.dtype, buffer=buffer_target)
            shared_target_names = np.ndarray(target_names.shape, dtype=target_names.dtype, buffer=buffer_target_names)

            shared_val_data = np.ndarray(val_data.shape, dtype=val_data.dtype, buffer=val_buffer)
            shared_val_target = np.ndarray(val_target.shape, dtype=val_target.dtype, buffer=val_buffer_target)
            shared_val_target_names = np.ndarray(val_target_names.shape, dtype=val_target_names.dtype,
                                                 buffer=val_buffer_target_names)

            logger.info("Generate shared data for each subprocess, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            shared_data[:] = data[:]
            shared_target[:] = target[:]
            shared_target_names[:] = target_names[:]

            shared_data_name = shared_memory_data.name
            shared_target_name = shared_memory_target.name
            shared_memory_target_names_name = shared_memory_target_names.name

            shared_val_data[:] = val_data[:]
            shared_val_target[:] = val_target[:]
            shared_val_target_names[:] = val_target_names[:]

            shared_val_data_name = shared_memory_val_data.name
            shared_val_target_name = shared_memory_val_target.name
            shared_memory_val_target_names_name = shared_memory_val_target_names.name

            pipeline_params = []
            logger.info("Generate params for each subprocess, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])

            for supervised_selector in self._supervised_selector_names:
                for auto_ml in self._auto_ml_names:
                    for model in self._model_names:
                        # 如果未进行数据清洗, 并且模型需要数据清洗, 则返回None.
                        if check_data(already_data_clear=self.already_data_clear, model_name=model) is True:
                            pipeline_params.append(
                                [feature_dict, work_root, supervised_feature_selector_flag, auto_ml_path,
                                 selector_config_path, model, supervised_selector, auto_ml, shared_data_name,
                                 data.shape, data.dtype, shared_target_name,
                                 target.shape, target.dtype, shared_memory_target_names_name, target_names.shape,
                                 target_names.dtype, shared_val_data_name, val_data.shape, val_data.dtype,
                                 shared_val_target_name, val_target.shape, val_target.dtype,
                                 shared_memory_val_target_names_name, val_target_names.shape, val_target_names.dtype])

            logger.info("Clearing additional object and save memory, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            del entity_dict, dataset, val_dataset, data, target, target_names, val_data, val_target, val_target_names
            gc.collect()

            self.jobs = min(len(pipeline_params), cpu_count())
            assert isinstance(self.jobs, int)
            logger.info("%d job(s) will be used for this task, with current memory usage: %.2f GiB",
                        self.jobs, get_current_memory_gb()["memory_usage"])

            logger.info("Create multi-preprocess and train dataset, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            with Pool(self.jobs) as pool:
                # make sure all subprocess finished.
                async_result = pool.map_async(func=self._single_run_route, iterable=pipeline_params,
                                              callback=multiprocess_callback)
                # Maximum training time: 10h
                async_result.wait(timeout=36000)
                # if not successful, value error will return.
                try:
                    if async_result.ready():
                        if async_result.successful():
                            logger.info("All subprocess has finished.")
                    else:
                        logger.info("Not all subprocess is ready()")
                except ValueError:
                    logger.info("Not all subprocess has finished successfully.")

                logger.info("All subprocess have been shut down." + "with current memory usage: %.2f GiB",
                            get_current_memory_gb()["memory_usage"])

            async_result = async_result.get(timeout=10)

            logger.info("Find best model and update pipeline configure, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            self.find_best_async_result(async_result)
        finally:

            logger.info("Remove and delete shared memory, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            shared_memory_data.unlink()
            shared_memory_target.unlink()
            shared_memory_target_names.unlink()
            shared_memory_val_data.unlink()
            shared_memory_val_target.unlink()
            shared_memory_val_target_names.unlink()

        logger.info("Multiprocess udf tree has finished.")

    def find_best_async_result(self, async_result):
        logger.info("Update best model from %d subprocess.", self.jobs)
        success_results = []
        model_names = set()

        best_result = {}
        for subprocess_result in async_result:

            assert isinstance(subprocess_result, dict)
            if subprocess_result.get("successful_flag"):

                if subprocess_result.get("successful_flag") is True:
                    success_results.append(subprocess_result)
                    model_names.add(subprocess_result.get("model_name"))

                else:
                    logger.info("A subprocess is not successful, async result: " + str(subprocess_result))

        if len(success_results) == 0:
            raise NoResultReturnException("All subprocesses have failed.")
        logger.info("%d subprocess(es) have return result(s) successfully.", len(success_results))

        for subprocess_result in success_results:
            model_name = subprocess_result.get("model_name")

            if best_result.get(model_name) is None:
                best_result[model_name] = subprocess_result

            else:
                if subprocess_result.get("metrics_result") is not None:
                    if best_result.get(model_name).get("metrics_result").__cmp__(
                            subprocess_result.get("metrics_result")) < 0:
                        best_result[model_name] = subprocess_result

        for subprocess_result in success_results:
            subprocess_result["metrics_result"] = float(subprocess_result.get("metrics_result").result)

        self.pipeline_config.update(best_result)

    def set_pipeline_config(self):

        yaml_dict = {}

        if self.pipeline_config is not None:
            yaml_dict.update(self.pipeline_config)

        yaml_write(yaml_dict=yaml_dict, yaml_file=self.work_root + "/pipeline_config.yaml")

    def _run(self):
        self.run_route(data_clear_flag=self.data_clear_flag,
                       feature_generator_flag=self.feature_generator_flag,
                       unsupervised_feature_selector_flag=self.unsupervised_feature_selector_flag,
                       supervised_feature_selector_flag=self.supervised_feature_selector_flag,
                       auto_ml_path=self.auto_ml_path,
                       selector_config_path=self.selector_config_path)

    def _preprocessing_run_route(self,
                                 feature_dict,
                                 data_clear_flag,
                                 feature_generator_flag,
                                 unsupervised_feature_selector_flag):

        assert isinstance(data_clear_flag, bool)
        assert isinstance(feature_generator_flag, bool)
        assert isinstance(unsupervised_feature_selector_flag, bool)

        preprocess_chain = PreprocessRoute(name="PreprocessRoute",
                                           feature_path_dict=feature_dict,
                                           task_type=self.task_type,
                                           train_flag=True,
                                           train_data_path=self.train_data_path,
                                           dataset_name=self.dataset_type,
                                           val_data_path=self.val_data_path,
                                           test_data_path=None,
                                           target_names=self.target_names,
                                           type_inference_name=self.type_inference,
                                           data_clear_name=self.data_clear,
                                           data_clear_flag=data_clear_flag,
                                           feature_generator_name=self.feature_generator,
                                           feature_generator_flag=feature_generator_flag,
                                           feature_selector_name=self.unsupervised_feature_selector,
                                           feature_selector_flag=unsupervised_feature_selector_flag)

        try:
            preprocess_chain.run()
        except PipeLineLogicError as e:
            logger.info(e)
            return None

        return preprocess_chain

    def _single_run_route(self, params):
        successful_flag = True
        entity_dict = dict()

        feature_dict = params[0]
        work_root = params[1]
        supervised_feature_selector_flag = params[2]
        auto_ml_path = params[3]
        selector_config_path = params[4]
        model_name = params[5]
        supervised_selector_name = params[6]
        auto_ml_name = params[7]

        logger.info("Loading dataset from shared memory, " + "current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        shared_data_memory = shared_memory.SharedMemory(name=params[8])
        shared_data = np.ndarray(params[9], dtype=params[10], buffer=shared_data_memory.buf)

        shared_target_memory = shared_memory.SharedMemory(name=params[11])
        shared_target = np.ndarray(params[12], dtype=params[13], buffer=shared_target_memory.buf)

        shared_target_names_memory = shared_memory.SharedMemory(name=params[14])
        shared_target_names = np.ndarray(params[15], dtype=params[16],
                                         buffer=shared_target_names_memory.buf).tolist()

        data_pair = Bunch(data=shared_data, target=shared_target, target_names=shared_target_names)
        entity_dict["dataset"] = MultiprocessPlaintextDataset(name="train_data", task_type=self.task_type,
                                                              data_pair=data_pair)

        shared_val_data_memory = shared_memory.SharedMemory(name=params[17])
        shared_val_data = np.ndarray(params[18], dtype=params[19], buffer=shared_val_data_memory.buf)

        shared_val_target_memory = shared_memory.SharedMemory(name=params[20])
        shared_val_target = np.ndarray(params[21], dtype=params[22], buffer=shared_val_target_memory.buf)

        shared_val_target_names_memory = shared_memory.SharedMemory(name=params[23])
        shared_val_target_names = np.ndarray(params[24], dtype=params[25],
                                             buffer=shared_val_target_names_memory.buf).tolist()

        data_pair = Bunch(data=shared_val_data, target=shared_val_target, target_names=shared_val_target_names)
        entity_dict["val_dataset"] = MultiprocessPlaintextDataset(name="val_data", task_type=self.task_type,
                                                                  data_pair=data_pair)

        work_model_root = work_root + "/model/" + model_name + "_" + auto_ml_name
        model_save_root = work_model_root + "/model_save"
        model_config_root = work_model_root + "/model_parameters"
        feature_config_root = work_model_root + "/feature_config"

        feature_dict["final_feature_config"] = feature_config_root + "/" + EnvironmentConfigure.feature_dict().final_feature_config

        local_metric = None

        try:
            logger.info("Initialize core route object, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])

            core_chain = CoreRoute(name="core_route",
                                   train_flag=True,
                                   model_save_root=model_save_root,
                                   model_config_root=model_config_root,
                                   feature_config_root=feature_config_root,
                                   target_feature_configure_path=feature_dict["final_feature_config"],
                                   pre_feature_configure_path=feature_dict["unsupervised_feature"],
                                   model_name=model_name,
                                   label_encoding_path=feature_dict["label_encoding_path"],
                                   metrics_name=self.metric_name,
                                   task_type=self.task_type,
                                   feature_selector_name=self.supervised_feature_selector,
                                   feature_selector_flag=supervised_feature_selector_flag,
                                   supervised_selector_name=supervised_selector_name,
                                   auto_ml_type=self.auto_ml,
                                   auto_ml_name=auto_ml_name,
                                   auto_ml_path=auto_ml_path,
                                   selector_config_path=selector_config_path)

            logger.info("Core route running for training model, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            core_chain.run(**entity_dict)

            local_metric = core_chain.optimal_metrics
            assert local_metric is not None

            logger.info("Close shared memory, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])

            logger.info(
                "Subprocess has finished, and waiting for other subprocess finished, " + "with current memory usage: %.2f GiB",
                get_current_memory_gb()["memory_usage"])

        except (Exception,):
            successful_flag = False

        finally:
            if successful_flag is True:
                return {"auto_ml_name": auto_ml_name,
                        "successful_flag": successful_flag,
                        "work_model_root": work_model_root,
                        "model_name": model_name,
                        "metrics_result": local_metric,
                        "final_file_path": feature_dict["final_feature_config"]}
            else:
                logger.info("Subprocess does not finished successfully.")
                return {"auto_ml_name": auto_ml_name,
                        "successful_flag": successful_flag,
                        "work_model_root": work_model_root,
                        "model_name": model_name}
