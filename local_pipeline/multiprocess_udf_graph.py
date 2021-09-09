"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
This pipeline is multiprocess version of udf modeling graph.
"""
from __future__ import annotations

import gc
from os.path import join
from multiprocessing import Pool, shared_memory, cpu_count

import numpy as np

from entity.dataset.multiprocess_plain_dataset import MultiprocessPlaintextDataset
from local_pipeline.core_chain import CoreRoute
from local_pipeline.preprocess_chain import PreprocessRoute
from local_pipeline.mapping import EnvironmentConfigure
from local_pipeline.base_modeling_graph import BaseModelingGraph
from utils.yaml_exec import yaml_write

from utils.exception import PipeLineLogicError, NoResultReturnException
from utils.base import get_current_memory_gb
from utils.check_dataset import check_data
from utils.Logger import logger
from utils.bunch import Bunch
from utils.callback import multiprocess_callback


# local_pipeline defined by user.
class MultiprocessUdfModelingGraph(BaseModelingGraph):
    """
    MultiprocessUdfModelingGraph object.
    """

    def __init__(self, name: str, **params):
        """
        :param name:
        :param work_root:
        :param task_name:
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
        super().__init__(
            name=name,
            work_root=params["work_root"],
            task_name=params["task_name"],
            metric_name=params["metric_name"],
            train_data_path=params["train_data_path"],
            val_data_path=params["val_data_path"],
            target_names=params["target_names"],
            feature_configure_path=params["feature_configure_path"],
            feature_configure_name=params["feature_configure_name"],
            dataset_name=params["dataset_name"],
            type_inference_name=params["type_inference_name"],
            label_encoder_name=params["label_encoder_name"],
            data_clear_name=params["data_clear_name"],
            feature_generator_name=params["feature_generator_name"],
            unsupervised_feature_selector_name=params["unsupervised_feature_selector_name"],
            supervised_feature_selector_name=params["supervised_feature_selector_name"],
            supervised_selector_model_names=params["supervised_selector_model_names"],
            selector_trial_num=params["selector_trial_num"],
            auto_ml_name=params["auto_ml_name"],
            auto_ml_trial_num=params["auto_ml_trial_num"],
            opt_model_names=params["opt_model_names"],
            auto_ml_path=params["auto_ml_path"],
            selector_config_path=params["selector_config_path"]
        )

        if params["model_zoo"] is None:
            params["model_zoo"] = ["xgboost", "lightgbm", "catboost", "lr_lightgbm", "dnn"]

        if params["supervised_feature_selector_flag"] is None:
            params["supervised_feature_selector_flag"] = True

        if params["unsupervised_feature_selector_flag"] is None:
            params["unsupervised_feature_selector_flag"] = True

        if params["feature_generator_flag"] is None:
            params["feature_generator_flag"] = True

        if params["data_clear_flag"] is None:
            params["data_clear_flag"] = True

        self._flag_dict = Bunch(
            data_clear_flag=params["data_clear_flag"],
            label_encoder_flag=params["label_encoder_name"],
            feature_generator_flag=params["feature_generator_flag"],
            unsupervised_feature_selector_flag=params["unsupervised_feature_selector_flag"],
            supervised_feature_selector_flag=params["supervised_feature_selector_flag"]
        )

        self._model_zoo = params["model_zoo"]

        self.jobs = None

        self.__pipeline_configure = \
            {"work_root":
                 self._work_paths["work_root"],
             "data_clear_flag":
                 self._flag_dict["data_clear_flag"],
             "data_clear_name":
                 self._component_names["data_clear_name"],
             "feature_generator_flag":
                 self._flag_dict["feature_generator_flag"],
             "feature_generator_name":
                 self._component_names["feature_generator_name"],
             "unsupervised_feature_selector_flag":
                 self._flag_dict["unsupervised_feature_selector_flag"],
             "unsupervised_feature_selector_name":
                 self._component_names["unsupervised_feature_selector_name"],
             "supervised_feature_selector_flag":
                 self._flag_dict["supervised_feature_selector_flag"],
             "supervised_feature_selector_name":
                 self._component_names["supervised_feature_selector_name"],
             "metric_name":
                 self._entity_names["metric_name"],
             "task_name":
                 self._attributes_names["task_name"],
             "target_names":
                 self._attributes_names["target_names"],
             "dataset_name":
                 self._entity_names["dataset_name"],
             "type_inference_name":
                 self._component_names["type_inference_name"]
             }

    def _run_route(self, **params):

        assert isinstance(self._flag_dict["data_clear_flag"], bool) and \
               isinstance(self._flag_dict["feature_generator_flag"], bool) and \
               isinstance(self._flag_dict["unsupervised_feature_selector_flag"], bool) and \
               isinstance(self._flag_dict["supervised_feature_selector_flag"], bool)

        work_root = self._work_paths["work_root"]

        work_feature_root = work_root + "/feature"

        feature_dict = EnvironmentConfigure.feature_dict()
        feature_dict = {"user_feature": self._work_paths[
            "feature_configure_path"
        ],
                        "type_inference_feature": join(
                            work_feature_root,
                            feature_dict.type_inference_feature),

                        "data_clear_feature": join(
                            work_feature_root,
                            feature_dict.data_clear_feature),

                        "feature_generator_feature": join(
                            work_feature_root,
                            feature_dict.feature_generator_feature),

                        "unsupervised_feature": join(
                            work_feature_root,
                            feature_dict.unsupervised_feature),

                        "supervised_feature": join(
                            work_feature_root,
                            feature_dict.supervised_feature),

                        "label_encoding_path": join(
                            work_feature_root,
                            feature_dict.label_encoding_path),

                        "impute_path": join(
                            work_feature_root,
                            feature_dict.impute_path),

                        "label_encoder_feature": join(
                            work_feature_root,
                            feature_dict.label_encoder_feature)
                        }

        preprocess_chain = self._preprocessing_run_route(feature_dict)

        entity_dict = preprocess_chain.entity_dict
        self._already_data_clear = preprocess_chain.already_data_clear

        assert "train_dataset" in entity_dict and "val_dataset" in entity_dict
        dataset = entity_dict["train_dataset"]
        val_dataset = entity_dict["val_dataset"]

        logger.info(
            "Loading shared memory for dataset, "
            "with current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        shared_memory_data = None
        shared_memory_target = None
        shared_memory_target_names = None
        shared_memory_val_data = None
        shared_memory_val_target = None
        shared_memory_val_target_names = None

        try:
            logger.info(
                "Read dataset, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            data = dataset.get_dataset().data.values
            target = dataset.get_dataset().target.values
            target_names = np.array(dataset.get_dataset().target_names)

            val_data = val_dataset.get_dataset().data.values
            val_target = val_dataset.get_dataset().target.values
            val_target_names = target_names

            logger.info(
                "Generate shared memory for each subprocess, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            shared_memory_data = shared_memory.SharedMemory(
                create=True,
                size=data.nbytes
            )
            shared_memory_target = shared_memory.SharedMemory(
                create=True,
                size=target.nbytes
            )
            shared_memory_target_names = shared_memory.SharedMemory(
                create=True,
                size=target_names.nbytes
            )

            shared_memory_val_data = shared_memory.SharedMemory(
                create=True,
                size=val_data.nbytes
            )
            shared_memory_val_target = shared_memory.SharedMemory(
                create=True,
                size=val_target.nbytes
            )
            shared_memory_val_target_names = shared_memory.SharedMemory(
                create=True,
                size=val_target_names.nbytes
            )

            logger.info(
                "Generate buffer for shared memory, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            buffer = shared_memory_data.buf
            buffer_target = shared_memory_target.buf
            buffer_target_names = shared_memory_target_names.buf

            val_buffer = shared_memory_val_data.buf
            val_buffer_target = shared_memory_val_target.buf
            val_buffer_target_names = shared_memory_val_target_names.buf

            logger.info(
                "Generate shared data buffer for each subprocess, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            shared_data = np.ndarray(
                data.shape,
                dtype=data.dtype,
                buffer=buffer
            )
            shared_target = np.ndarray(
                target.shape,
                dtype=target.dtype,
                buffer=buffer_target
            )
            shared_target_names = np.ndarray(
                target_names.shape,
                dtype=target_names.dtype,
                buffer=buffer_target_names
            )

            shared_val_data = np.ndarray(
                val_data.shape,
                dtype=val_data.dtype,
                buffer=val_buffer
            )
            shared_val_target = np.ndarray(
                val_target.shape,
                dtype=val_target.dtype,
                buffer=val_buffer_target
            )
            shared_val_target_names = np.ndarray(
                val_target_names.shape,
                dtype=val_target_names.dtype,
                buffer=val_buffer_target_names
            )

            logger.info(
                "Generate shared data for each subprocess, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
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
            logger.info(
                "Generate params for each subprocess, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            for selector_model_name in self._global_values["supervised_selector_model_names"]:
                for opt_model_name in self._global_values["opt_model_names"]:
                    for model_name in self._model_zoo:
                        # 如果未进行数据清洗, 并且模型需要数据清洗, 则返回None.
                        if check_data(
                                already_data_clear=self._already_data_clear,
                                model_name=model_name
                        ) is True:
                            pipeline_params.append(
                                {
                                    "feature_dict": feature_dict,
                                    "work_root": work_root,
                                    "supervised_feature_selector_flag": self._flag_dict[
                                        "supervised_feature_selector_flag"
                                    ],
                                    "auto_ml_path": self._work_paths["auto_ml_path"],
                                    "selector_config_path": self._work_paths[
                                        "selector_config_path"
                                    ],
                                    "model_name": model_name,
                                    "selector_model_name": [selector_model_name],
                                    "opt_model_name": [opt_model_name],
                                    "train_data_attributes":
                                        {
                                            "name": shared_data_name,
                                            "shape": shared_data.shape,
                                            "dtype": shared_data.dtype
                                        },
                                    "train_target_attributes":
                                        {
                                            "name": shared_target_name,
                                            "shape": shared_target.shape,
                                            "dtype": shared_target.dtype
                                        },
                                    "train_target_names_attributes":
                                        {
                                            "name": shared_memory_target_names_name,
                                            "shape": shared_target_names.shape,
                                            "dtype": shared_target_names.dtype
                                        },
                                    "val_data_attributes":
                                        {
                                            "name": shared_val_data_name,
                                            "shape": shared_val_data.shape,
                                            "dtype": shared_val_data.dtype
                                        },
                                    "val_target_attributes":
                                        {
                                            "name": shared_val_target_name,
                                            "shape": shared_val_target.shape,
                                            "dtype": shared_val_target.dtype
                                        },
                                    "val_target_names_attributes":
                                        {
                                            "name": shared_memory_val_target_names_name,
                                            "shape": shared_val_target_names.shape,
                                            "dtype": shared_val_target_names.dtype
                                        },
                                    "feature_configure_name": self._entity_names[
                                        "feature_configure_name"
                                    ],
                                    "selector_trial_num": self._global_values["selector_trial_num"],
                                    "auto_ml_trial_num": self._global_values["auto_ml_trial_num"],
                                    "supervised_feature_selector_name": self._component_names[
                                        "supervised_feature_selector_name"],
                                    "auto_ml_name": self._component_names["auto_ml_name"]
                                }
                            )

            logger.info(
                "Clearing additional object and save memory, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            del entity_dict, \
                dataset, \
                val_dataset, \
                data, \
                target, \
                target_names, \
                val_data, \
                val_target, \
                val_target_names

            gc.collect()

            self.jobs = min(len(pipeline_params), cpu_count())
            assert isinstance(self.jobs, int)
            logger.info("%d job(s) will be used for this task, "
                        "with current memory usage: %.2f GiB",
                        self.jobs, get_current_memory_gb()["memory_usage"])

            logger.info(
                "Create multi-preprocess and train dataset, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
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

                logger.info(
                    "All subprocess have been shut down, "
                    "with current memory usage: {:.2f} GiB".format(
                        get_current_memory_gb()["memory_usage"]
                    )
                )

            async_result = async_result.get(timeout=10)

            logger.info(
                "Find best model and update local_pipeline configure, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            self._find_best_result(async_result)
        finally:

            logger.info(
                "Remove and delete shared memory, with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            shared_memory_data.unlink()
            shared_memory_target.unlink()
            shared_memory_target_names.unlink()
            shared_memory_val_data.unlink()
            shared_memory_val_target.unlink()
            shared_memory_val_target_names.unlink()

        logger.info("Multiprocess udf tree has finished.")

    def _find_best_result(self, train_results):
        logger.info("Update best model from {:d} subprocess.".format(self.jobs))
        success_results = []
        model_names = set()

        best_result = {}
        for subprocess_result in train_results:

            assert isinstance(subprocess_result, dict)
            if subprocess_result.get("successful_flag"):

                if subprocess_result.get("successful_flag") is True:
                    success_results.append(subprocess_result)
                    model_names.add(subprocess_result.get("model_name"))

        if len(success_results) == 0:
            raise NoResultReturnException("All subprocesses have failed.")
        logger.info(
            "{:d} subprocess(es) have return result(s) successfully.".format(
                len(success_results)
            )
        )

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
            result = subprocess_result.get("metrics_result").result
            subprocess_result["metrics_result"] = float(result)

        self.__pipeline_configure.update(best_result)

    def _set_pipeline_config(self):

        yaml_dict = {}

        if self.__pipeline_configure is not None:
            yaml_dict.update(self.__pipeline_configure)

        yaml_write(yaml_dict=yaml_dict,
                   yaml_file=self._work_paths["work_root"] + "/pipeline_config.yaml")

    def _run(self):
        self._run_route()

    def _preprocessing_run_route(self, feature_dict):

        assert isinstance(self._flag_dict["data_clear_flag"], bool)
        assert isinstance(self._flag_dict["feature_generator_flag"], bool)
        assert isinstance(self._flag_dict["unsupervised_feature_selector_flag"], bool)

        preprocessing_params = Bunch(
            name="PreprocessRoute",
            feature_path_dict=feature_dict,
            task_name=self._attributes_names["task_name"],
            train_flag=True,
            train_data_path=self._work_paths["train_data_path"],
            dataset_name=self._entity_names["dataset_name"],
            val_data_path=self._work_paths["val_data_path"],
            test_data_path=None,
            target_names=self._attributes_names["target_names"],
            type_inference_name=self._component_names["type_inference_name"],
            label_encoder_name=self._component_names["label_encoder_name"],
            label_encoder_flag=self._flag_dict["label_encoder_flag"],
            data_clear_name=self._component_names["data_clear_name"],
            data_clear_flag=self._flag_dict["data_clear_flag"],
            feature_generator_name=self._component_names["feature_generator_name"],
            feature_generator_flag=self._flag_dict["feature_generator_flag"],
            unsupervised_feature_selector_name=self._component_names[
                "unsupervised_feature_selector_name"
            ],
            unsupervised_feature_selector_flag=self._flag_dict[
                "unsupervised_feature_selector_flag"
            ]
        )

        preprocess_chain = PreprocessRoute(**preprocessing_params)

        try:
            preprocess_chain.run()
        except PipeLineLogicError as error:
            logger.info(error)
            return None

        return preprocess_chain

    def _single_run_route(self, params):

        successful_flag = True
        entity_dict = {}

        feature_dict = params["feature_dict"]

        logger.info(
            "Loading dataset from shared memory, "
            "current memory usage: {:.2f} GiB".format(
                get_current_memory_gb()["memory_usage"]
            )
        )

        shared_data_memory = shared_memory.SharedMemory(
            name=params["train_data_attributes"]["name"]
        )
        shared_data = np.ndarray(
            shape=params["train_data_attributes"]["shape"],
            dtype=params["train_data_attributes"]["dtype"],
            buffer=shared_data_memory.buf
        )

        shared_target_memory = shared_memory.SharedMemory(
            name=params["train_target_attributes"]["name"]
        )
        shared_target = np.ndarray(
            shape=params["train_target_attributes"]["shape"],
            dtype=params["train_target_attributes"]["dtype"],
            buffer=shared_target_memory.buf
        )

        shared_target_names_memory = shared_memory.SharedMemory(
            name=params["train_target_names_attributes"]["name"]
        )
        shared_target_names = np.ndarray(
            shape=params["train_target_names_attributes"]["shape"],
            dtype=params["train_target_names_attributes"]["dtype"],
            buffer=shared_target_names_memory.buf
        ).tolist()

        entity_dict["train_dataset"] = MultiprocessPlaintextDataset(
            name="train_data",
            task_name=self._attributes_names["task_name"],
            data_pair=Bunch(
                data=shared_data,
                target=shared_target,
                target_names=shared_target_names
            )
        )

        shared_val_data_memory = shared_memory.SharedMemory(
            name=params["val_data_attributes"]["name"]
        )
        shared_val_data = np.ndarray(
            shape=params["val_data_attributes"]["shape"],
            dtype=params["val_data_attributes"]["dtype"],
            buffer=shared_val_data_memory.buf
        )

        shared_val_target_memory = shared_memory.SharedMemory(
            name=params["val_target_attributes"]["name"]
        )
        shared_val_target = np.ndarray(
            shape=params["val_target_attributes"]["shape"],
            dtype=params["val_target_attributes"]["dtype"],
            buffer=shared_val_target_memory.buf
        )

        shared_val_target_names_memory = shared_memory.SharedMemory(
            name=params["val_target_names_attributes"]["name"]
        )
        shared_val_target_names = np.ndarray(
            shape=params["val_target_names_attributes"]["shape"],
            dtype=params["val_target_names_attributes"]["dtype"],
            buffer=shared_val_target_names_memory.buf
        ).tolist()

        entity_dict["val_dataset"] = MultiprocessPlaintextDataset(
            name="val_data",
            task_name=self._attributes_names["task_name"],
            data_pair=Bunch(
                data=shared_val_data,
                target=shared_val_target,
                target_names=shared_val_target_names
            )
        )

        work_model_root = params["work_root"] + "/model/" \
                          + params["model_name"] \
                          + "_" + params["auto_ml_name"]
        model_save_root = join(work_model_root, "model_save")
        model_config_root = join(work_model_root, "model_parameters")
        feature_config_root = join(work_model_root, "feature_config")

        feature_dict["final_feature_config"] = join(
            feature_config_root,
            EnvironmentConfigure.feature_dict().final_feature_config
        )

        local_metric = None

        try:
            logger.info(
                "Initialize core route object, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            core_params = Bunch(
                name="core_route",
                train_flag=True,
                model_name=params["model_name"],
                feature_configure_name=params["feature_configure_name"],
                model_save_root=model_save_root,
                model_config_root=model_config_root,
                feature_config_root=feature_config_root,
                target_feature_configure_path=feature_dict["final_feature_config"],
                pre_feature_configure_path=feature_dict["unsupervised_feature"],
                label_encoding_path=feature_dict["label_encoding_path"],
                metrics_name=self._entity_names["metric_name"],
                task_name=self._attributes_names["task_name"],
                feature_selector_model_names=params["selector_model_name"],
                selector_trial_num=params["selector_trial_num"],
                supervised_feature_selector_flag=params["supervised_feature_selector_flag"],
                supervised_selector_name=params["supervised_feature_selector_name"],
                auto_ml_path=params["auto_ml_path"],
                auto_ml_name=params["auto_ml_name"],
                auto_ml_trial_num=params["auto_ml_trial_num"],
                opt_model_names=params["opt_model_name"],
                selector_config_path=params["selector_config_path"]
            )

            core_chain = CoreRoute(**core_params)

            logger.info(
                "Core route running for training model, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )
            core_chain.run(**entity_dict)

            local_metric = core_chain.optimal_metrics
            assert local_metric is not None

            logger.info(
                "Close shared memory, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

            logger.info(
                "Subprocess has finished, "
                "and waiting for other subprocess finished, "
                "with current memory usage: {:.2f} GiB".format(
                    get_current_memory_gb()["memory_usage"]
                )
            )

        except (Exception,):
            successful_flag = False

        finally:
            if successful_flag is True:
                subprocess_result = {"auto_ml_name": params["auto_ml_name"],
                                     "successful_flag": successful_flag,
                                     "work_model_root": work_model_root,
                                     "model_name": params["model_name"],
                                     "metrics_result": local_metric,
                                     "final_file_path": feature_dict["final_feature_config"]}
            else:
                logger.info("Subprocess does not finished successfully.")
                subprocess_result = {"auto_ml_name": params["auto_ml_name"],
                                     "successful_flag": successful_flag,
                                     "work_model_root": work_model_root,
                                     "model_name": params["model_name"]}
        return subprocess_result
