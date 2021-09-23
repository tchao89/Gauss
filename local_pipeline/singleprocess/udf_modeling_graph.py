"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
This pipeline is used to train model, which parameters and settings can be customized by user.
"""
from __future__ import annotations

from os.path import join

from local_pipeline.sub_pipeline.core_chain import CoreRoute
from local_pipeline.sub_pipeline.preprocess_chain import PreprocessRoute
from local_pipeline.pipeline_utils.mapping import EnvironmentConfigure
from local_pipeline.base_modeling_graph import BaseModelingGraph

from utils.check_dataset import check_data
from utils.yaml_exec import yaml_write
from utils.exception import PipeLineLogicError, NoResultReturnException
from utils.Logger import logger
from utils.constant_values import ConstantValues


# local_pipeline defined by user.
class UdfModelingGraph(BaseModelingGraph):
    """
    UdfModelingGraph object.
    In this pipeline, value: train_flag will be set "train"
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
        super().__init__(
            name=name,
            data_file_type=params[ConstantValues.data_file_type],
            work_root=params[ConstantValues.work_root],
            task_name=params[ConstantValues.task_name],
            metric_name=params[ConstantValues.metric_name],
            loss_name=params[ConstantValues.loss_name],
            train_data_path=params[ConstantValues.train_data_path],
            val_data_path=params[ConstantValues.val_data_path],
            target_names=params[ConstantValues.target_names],
            model_need_clear_flag=params[ConstantValues.model_need_clear_flag],
            feature_configure_path=params[ConstantValues.feature_configure_path],
            feature_configure_name=params[ConstantValues.feature_configure_name],
            dataset_name=params[ConstantValues.dataset_name],
            type_inference_name=params[ConstantValues.type_inference_name],
            label_encoder_name=params[ConstantValues.label_encoder_name],
            label_encoder_flag=params[ConstantValues.label_encoder_flag],
            data_clear_name=params[ConstantValues.data_clear_name],
            data_clear_flag=params[ConstantValues.data_clear_flag],
            feature_generator_name=params[ConstantValues.feature_generator_name],
            feature_generator_flag=params[ConstantValues.feature_generator_flag],
            unsupervised_feature_selector_name=params[ConstantValues.unsupervised_feature_selector_name],
            unsupervised_feature_selector_flag=params[ConstantValues.unsupervised_feature_selector_flag],
            supervised_feature_selector_name=params[ConstantValues.supervised_feature_selector_name],
            supervised_feature_selector_flag=params[ConstantValues.supervised_feature_selector_flag],
            supervised_selector_model_names=params[ConstantValues.supervised_selector_model_names],
            selector_trial_num=params[ConstantValues.selector_trial_num],
            auto_ml_name=params[ConstantValues.auto_ml_name],
            auto_ml_trial_num=params[ConstantValues.auto_ml_trial_num],
            opt_model_names=params[ConstantValues.opt_model_names],
            auto_ml_path=params[ConstantValues.auto_ml_path],
            selector_configure_path=params[ConstantValues.selector_configure_path]
        )

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

    def __path_register(self):
        pass

    def _run_route(self, **params):
        assert isinstance(self._flag_dict[ConstantValues.data_clear_flag], bool) and \
               isinstance(self._flag_dict[ConstantValues.feature_generator_flag], bool) and \
               isinstance(self._flag_dict[ConstantValues.unsupervised_feature_selector_flag], bool) and \
               isinstance(self._flag_dict[ConstantValues.supervised_feature_selector_flag], bool)

        work_feature_root = join(self._work_paths[ConstantValues.work_root], ConstantValues.feature)
        feature_dict = EnvironmentConfigure.feature_dict()

        feature_dict = \
            {ConstantValues.user_feature_path: self._work_paths[ConstantValues.feature_configure_path],
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
            join(self._work_paths[ConstantValues.work_root], ConstantValues.model),
            params.get(ConstantValues.model_name)
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
            task_name=self._attributes_names[ConstantValues.task_name],
            train_flag=ConstantValues.train,
            train_data_path=self._work_paths[ConstantValues.train_data_path],
            val_data_path=self._work_paths[ConstantValues.val_data_path],
            test_data_path=None,
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
            preprocess_chain.run()
        except PipeLineLogicError as error:
            logger.info(error)
            return None

        entity_dict = preprocess_chain.entity_dict
        self._already_data_clear = preprocess_chain.already_data_clear

        assert params.get(ConstantValues.model_name) is not None
        # 如果未进行数据清洗, 并且模型需要数据清洗, 则返回None.
        model_name = (params.get(ConstantValues.model_name))
        if check_data(already_data_clear=self._already_data_clear,
                      model_need_clear_flag=self._model_need_clear_flag.get(model_name)) is not True:
            return None

        assert ConstantValues.train_dataset in entity_dict and ConstantValues.val_dataset in entity_dict

        core_chain = CoreRoute(
            name=ConstantValues.CoreRoute,
            train_flag=ConstantValues.train,
            model_root_path=work_model_root,
            target_feature_configure_path=feature_dict[ConstantValues.final_feature_configure],
            pre_feature_configure_path=feature_dict[ConstantValues.unsupervised_feature_path],
            model_name=params.get(ConstantValues.model_name),
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
        local_metric = core_chain.optimal_metric
        assert local_metric is not None
        return {"work_model_root": work_model_root,
                "model_name": params.get(ConstantValues.model_name),
                "metric_result": local_metric,
                "final_file_path": feature_dict[ConstantValues.final_feature_configure]}

    def _run(self):
        train_results = []

        for model in self._model_zoo:
            local_result = self._run_route(model_name=model)

            if local_result is not None:
                train_results.append(local_result)
            self._find_best_result(train_results=train_results)

    def _find_best_result(self, train_results):

        best_result = {}

        if len(train_results) == 0:
            raise NoResultReturnException("No model is trained successfully.")

        for result in train_results:
            model_name = result.get(ConstantValues.model_name)

            if best_result.get(model_name) is None:
                best_result[model_name] = result

            else:
                if result.get(ConstantValues.metric_result) is not None:
                    if best_result.get(model_name).get(ConstantValues.metric_result).__cmp__(
                            result.get(ConstantValues.metric_result)) < 0:
                        best_result[model_name] = result

        for result in train_results:
            result[ConstantValues.metric_result] = float(result.get(ConstantValues.metric_result).result)

        self.__pipeline_configure.update(best_result)

    def _set_pipeline_config(self):
        feature_dict = EnvironmentConfigure.feature_dict()
        yaml_dict = {}

        if self.__pipeline_configure is not None:
            yaml_dict.update(self.__pipeline_configure)

        yaml_write(yaml_dict=yaml_dict,
                   yaml_file=join(self._work_paths["work_root"], feature_dict.pipeline_configure))

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
