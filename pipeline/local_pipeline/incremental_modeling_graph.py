"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
increment pipeline.
"""
from __future__ import annotations

from os.path import join

from pipeline.local_pipeline.core_chain import CoreRoute
from pipeline.local_pipeline.preprocess_chain import PreprocessRoute
from pipeline.local_pipeline.mapping import EnvironmentConfigure
from utils.bunch import Bunch

from utils.yaml_exec import yaml_write
from utils.exception import PipeLineLogicError, NoResultReturnException
from utils.Logger import logger
from utils.constant_values import ConstantValues


class IncrementModelingGraph:
    """
    Increment object.
    In this pipeline, value: train_flag will be set "increment"
    between "train", "inference" and "increment".
    """

    def __init__(self, name: str, **params):
        """
        :param name: string project, pipeline name
        :param work_root: project work root
        :param task_name:
        :param metric_name:
        :param train_data_path:
        :param val_data_path:
        :param target_names:
        :param feature_configure_path:
        :param dataset_name:
        :param type_inference_name:
        :param data_clear_name:
        :param data_clear_flag:
        :param feature_generator_name:
        :param feature_generator_flag:
        :param unsupervised_feature_selector_name:
        :param unsupervised_feature_selector_flag:
        :param supervised_feature_selector_name:
        :param supervised_feature_selector_flag:
        :param model_zoo:
        :param supervised_selector_model_names:
        :param auto_ml_name:
        :param opt_model_names:
        :param auto_ml_path:
        :param selector_configure_path:
        """
        self._name = name

        if params[ConstantValues.model_zoo] is None:
            params[ConstantValues.model_zoo] = ["xgboost", "lightgbm", "catboost", "lr_lightgbm", "dnn"]

        if params[ConstantValues.supervised_feature_selector_flag] is None:
            params[ConstantValues.supervised_feature_selector_flag] = True

        if params[ConstantValues.unsupervised_feature_selector_flag] is None:
            params[ConstantValues.unsupervised_feature_selector_flag] = True

        if params[ConstantValues.feature_generator_flag] is None:
            params[ConstantValues.feature_generator_flag] = True

        if params[ConstantValues.data_clear_flag] is None:
            params[ConstantValues.data_clear_flag] = True

        self._attributes_names = Bunch(
            task_name=params[ConstantValues.task_name],
            target_names=params[ConstantValues.target_names],
            metric_eval_used_flag=params[ConstantValues.metric_eval_used_flag]
        )

        self._work_paths = Bunch(
            work_root=params[ConstantValues.work_root],
            train_data_path=params[ConstantValues.train_data_path],
            val_data_path=params[ConstantValues.val_data_path],
            auto_ml_path=params[ConstantValues.auto_ml_path],
            selector_configure_path=params[ConstantValues.selector_configure_path]
        )

        self._entity_names = Bunch(
            dataset_name=params[ConstantValues.dataset_name],
            metric_name=params[ConstantValues.metric_name],
            loss_name=params[ConstantValues.loss_name],
            feature_configure_name=params[ConstantValues.feature_configure_name]
        )

        self._component_names = Bunch(
            type_inference_name=params[ConstantValues.type_inference_name],
            data_clear_name=params[ConstantValues.data_clear_name],
            label_encoder_name=params[ConstantValues.label_encoder_name],
            feature_generator_name=params[ConstantValues.feature_generator_name],
            unsupervised_feature_selector_name=params["unsupervised_feature_selector_name"],
            supervised_feature_selector_name=params["supervised_feature_selector_name"],
            improved_supervised_feature_selector_name=params["improved_supervised_feature_selector_name"],
            auto_ml_name=params["auto_ml_name"]
        )

        self._global_values = Bunch(
            use_weight_flag=params["use_weight_flag"],
            weight_column_name=params["weight_column_name"],
            decay_rate=params[ConstantValues.decay_rate],
            increment_column_name_flag=params[ConstantValues.increment_column_name_flag],
            data_file_type=params["data_file_type"],
            selector_trial_num=params["selector_trial_num"],
            auto_ml_trial_num=params["auto_ml_trial_num"],
            opt_model_names=params["opt_model_names"],
            supervised_selector_mode=params["supervised_selector_mode"],
            feature_model_trial=params["feature_model_trial"],
            supervised_selector_model_names=params["supervised_selector_model_names"]
        )

        self._flag_dict = Bunch(
            data_clear_flag=params["data_clear_flag"],
            label_encoder_flag=params["label_encoder_flag"],
            feature_generator_flag=params["feature_generator_flag"],
            unsupervised_feature_selector_flag=params["unsupervised_feature_selector_flag"],
            supervised_feature_selector_flag=params["supervised_feature_selector_flag"]
        )
        self._model_zoo = params[ConstantValues.model_zoo]

        self.__pipeline_configure = \
            {ConstantValues.work_root:
                 self._work_paths[ConstantValues.work_root],
             ConstantValues.data_clear_flag:
                 self._flag_dict[ConstantValues.data_clear_flag],
             ConstantValues.data_clear_name:
                 self._component_names[ConstantValues.data_clear_name],
             ConstantValues.feature_generator_flag:
                 self._flag_dict[ConstantValues.feature_generator_flag],
             ConstantValues.feature_generator_name:
                 self._component_names[ConstantValues.feature_generator_name],
             ConstantValues.unsupervised_feature_selector_flag:
                 self._flag_dict[ConstantValues.unsupervised_feature_selector_flag],
             ConstantValues.unsupervised_feature_selector_name:
                 self._component_names[ConstantValues.unsupervised_feature_selector_name],
             ConstantValues.supervised_feature_selector_flag:
                 self._flag_dict[ConstantValues.supervised_feature_selector_flag],
             ConstantValues.supervised_feature_selector_name:
                 self._component_names[ConstantValues.supervised_feature_selector_name],
             ConstantValues.metric_name:
                 self._entity_names[ConstantValues.metric_name],
             ConstantValues.task_name:
                 self._attributes_names[ConstantValues.task_name],
             ConstantValues.target_names:
                 self._attributes_names[ConstantValues.target_names],
             ConstantValues.dataset_name:
                 self._entity_names[ConstantValues.dataset_name],
             ConstantValues.type_inference_name:
                 self._component_names[ConstantValues.type_inference_name]
             }

    def run(self):
        """
        Start training model with pipeline.
        :return:
        """
        self._run()
        self._set_pipeline_config()

    def __path_register(self):
        pass

    def _run_route(self, **params):
        assert isinstance(self._flag_dict[ConstantValues.data_clear_flag], bool) and \
               isinstance(self._flag_dict[ConstantValues.feature_generator_flag], bool) and \
               isinstance(self._flag_dict[ConstantValues.unsupervised_feature_selector_flag], bool) and \
               isinstance(self._flag_dict[ConstantValues.supervised_feature_selector_flag], bool)

        dispatch_model_root = join(self._work_paths[ConstantValues.work_root], params[ConstantValues.model_name])
        work_feature_root = join(dispatch_model_root, ConstantValues.feature)
        feature_dict = EnvironmentConfigure.feature_dict()

        feature_dict = \
            {ConstantValues.user_feature_path: join(work_feature_root,
                                                    feature_dict.user_feature),
             ConstantValues.type_inference_feature_path: join(
                 work_feature_root,
                 feature_dict.type_inference_feature),

             ConstantValues.data_clear_feature_path: join(
                 work_feature_root,
                 feature_dict.data_clear_feature),

             ConstantValues.feature_generator_feature_path: join(
                 work_feature_root,
                 feature_dict.feature_generator_feature),

             ConstantValues.unsupervised_feature_path: join(
                 work_feature_root,
                 feature_dict.unsupervised_feature),

             ConstantValues.supervised_feature_path: join(
                 work_feature_root,
                 feature_dict.supervised_feature),

             ConstantValues.label_encoding_models_path: join(
                 work_feature_root,
                 feature_dict.label_encoding_path),

             ConstantValues.impute_models_path: join(
                 work_feature_root,
                 feature_dict.impute_path),

             ConstantValues.label_encoder_feature_path: join(
                 work_feature_root,
                 feature_dict.label_encoder_feature)
             }

        work_model_root = join(
            dispatch_model_root,
            ConstantValues.model
        )

        feature_configure_root = join(work_model_root, ConstantValues.feature_configure)
        feature_dict[ConstantValues.final_feature_configure] = join(
            feature_configure_root,
            EnvironmentConfigure.feature_dict().final_feature_configure
        )

        preprocess_chain = PreprocessRoute(
            name=ConstantValues.PreprocessRoute,
            feature_path_dict=feature_dict,
            data_file_type=self._global_values[ConstantValues.data_file_type],
            increment_column_name_flag=self._global_values[ConstantValues.increment_column_name_flag],
            task_name=self._attributes_names[ConstantValues.task_name],
            train_flag=ConstantValues.increment,
            train_data_path=self._work_paths[ConstantValues.train_data_path],
            val_data_path=None,
            inference_data_path=None,
            target_names=self._attributes_names[ConstantValues.target_names],
            dataset_name=self._entity_names[ConstantValues.dataset_name],
            type_inference_name=self._component_names[ConstantValues.type_inference_name],
            data_clear_name=self._component_names[ConstantValues.data_clear_name],
            data_clear_flag=self._flag_dict[ConstantValues.data_clear_flag],
            label_encoder_name=self._component_names[ConstantValues.label_encoder_name],
            label_encoder_flag=self._flag_dict[ConstantValues.label_encoder_flag],
            feature_generator_name=self._component_names[ConstantValues.feature_generator_name],
            feature_generator_flag=self._flag_dict[ConstantValues.feature_generator_flag],
            unsupervised_feature_selector_name=self._component_names[ConstantValues.unsupervised_feature_selector_name],
            unsupervised_feature_selector_flag=self._flag_dict[ConstantValues.unsupervised_feature_selector_flag]
        )

        try:
            entity_dict = preprocess_chain.run()
        except PipeLineLogicError as error:
            logger.info(error)
            return None

        self._already_data_clear = preprocess_chain.already_data_clear

        assert ConstantValues.increment_dataset in entity_dict
        decay_rate = params[ConstantValues.decay_rate]

        core_chain = CoreRoute(
            name=ConstantValues.CoreRoute,
            train_flag=ConstantValues.increment,
            model_root_path=work_model_root,
            decay_rate=decay_rate,
            target_feature_configure_path=feature_dict[ConstantValues.final_feature_configure],
            pre_feature_configure_path=feature_dict[ConstantValues.unsupervised_feature_path],
            model_name=params.get(ConstantValues.model_name),
            metric_eval_used_flag=self._attributes_names.metric_eval_used_flag,
            feature_configure_name=self._entity_names[ConstantValues.feature_configure_name],
            label_encoding_path=feature_dict[ConstantValues.label_encoding_models_path],
            metric_name=self._entity_names[ConstantValues.metric_name],
            loss_name=self._entity_names[ConstantValues.loss_name],
            task_name=self._attributes_names[ConstantValues.task_name],
            supervised_selector_name=self._component_names[ConstantValues.supervised_feature_selector_name],
            feature_selector_model_names=self._global_values[ConstantValues.supervised_selector_model_names],
            selector_trial_num=self._global_values[ConstantValues.selector_trial_num],
            supervised_feature_selector_flag=self._flag_dict[ConstantValues.supervised_feature_selector_flag],
            auto_ml_name=self._component_names[ConstantValues.auto_ml_name],
            auto_ml_trial_num=self._global_values[ConstantValues.auto_ml_trial_num],
            auto_ml_path=self._work_paths[ConstantValues.auto_ml_path],
            opt_model_names=self._global_values[ConstantValues.opt_model_names],
            selector_configure_path=self._work_paths[ConstantValues.selector_configure_path]
        )

        core_chain.run(**entity_dict)
        return {"work_model_root": work_model_root,
                ConstantValues.increment_flag: True,
                "model_name": params.get(ConstantValues.model_name),
                "final_file_path": feature_dict[ConstantValues.final_feature_configure]}

    def _run(self):
        assert isinstance(self._model_zoo, list)
        if len(self._model_zoo) == 0:
            raise ValueError("Value: model_zoo is empty.")
        if len(self._model_zoo) > 1:
            raise ValueError("Value: model_zoo can not contain more than one model name.")

        for model_name in self._model_zoo:
            decay_rate = self._global_values[ConstantValues.decay_rate][model_name]
            local_result = self._run_route(model_name=model_name, decay_rate=decay_rate)
            self.__pipeline_configure.update({model_name: local_result})

    def _set_pipeline_config(self):
        feature_dict = EnvironmentConfigure.feature_dict()
        yaml_dict = {}

        if self.__pipeline_configure is not None:
            yaml_dict.update(self.__pipeline_configure)

        yaml_write(yaml_dict=yaml_dict,
                   yaml_file=join(self._work_paths["work_root"], feature_dict.pipeline_configure))

    @classmethod
    def _find_best_result(cls, train_results):
        if len(train_results) == 0:
            raise NoResultReturnException("No model is trained successfully.")

    @property
    def pipeline_configure(self):
        """
        property method
        :return: A dict of udf model graph configuration.
        """
        if self.__pipeline_configure is None:
            raise RuntimeError("This pipeline has not start.")
        return self.__pipeline_configure

    @classmethod
    def build_pipeline(cls, name: str, **params):
        if name == ConstantValues.PreprocessRoute:
            return PreprocessRoute(**params)
        if name == ConstantValues.CoreRoute:
            return CoreRoute(**params)
