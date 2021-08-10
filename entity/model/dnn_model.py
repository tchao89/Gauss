# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab
# 
import os
import abc
import tensorflow as tf

from collections import deque

from entity.model.model import ModelWrapper
from core.tfdnn.trainers.trainer import Trainer
from core.tfdnn.evaluators.evaluator import Evaluator
from core.tfdnn.networks.mlp_network import MlpNetwork
from core.tfdnn.model_savers.model_saver import ModelSaver
from core.tfdnn.transforms.numerical_transform import NumericalTransform
from core.tfdnn.transforms.categorical_transform import CategoricalTransform
from core.tfdnn.statistics_gens.dataset_statistics_gen import DatasetStatisticsGen
from core.tfdnn.statistics_gens.external_statistics_gen import ExternalStatisticsGen
from utils.utils import tf_global_config
from utils.common_component import mkdir
from utils.common_component import yaml_read
from gauss_factory.loss_factory import LossFunctionFactory


class GuassNN(ModelWrapper):

    def __init__(self, **params):
        """"""
        super(GuassNN, self).__init__(
            name=params["name"],
            model_path=params["model_path"],
            model_config_root=params["model_config_root"],
            feature_config_root=params["feature_config_root"],
            task_type=params["task_type"],
            train_flag=params["train_flag"]
            )

        self.model_file_name = params["model_root"] + "/" + self.name + ".txt"
        self.model_config_file_name = params["model_config_root"] + "/" + self.name + ".model_conf.yaml"
        self.feature_config_file_name = params["feature_config_root"] + "/" + self.name + ".final.yaml"

        self._categorical_features = None
        self._best_categorical_features = None
        self._numerical_features = None
        self._best_numerical_features = None
        self._feature_statistics_filepath = self._model_root + "/.statistics"
        self._save_checkpoints_dir = mkdir(self._model_root + "/checkpoint/")
        self._restore_checkpoint_path = mkdir(self._model_root + "/restore_checkpoint/")
        self._tensorboard_logdir = mkdir(self._model_root + "/tensorboard_logdir/")
        self._save_model_dir = self._model_root + "/saved_model"

        self._statistics_gen = None
        self._best_statistics_gen = None
        self._statistics = None
        self._best_statistics = None
        self._transform1 = None
        self._best_transform1 = None
        self._transform2 = None
        self._best_transform2 = None
        self._network = None
        self._best_network = None
        self._evaluator = None
        self._best_evaluator = None
        self._trainer = None
        self._best_trainer = None
        tf_global_config(intra_threads=8, inter_threads=8)


    def _update_best_modules(self):
        self._best_statistics_gen = self._statistics_gen
        self._best_statistics = self._statistics
        self._best_transform1 = self._transform1
        self._best_transform2 = self._transform2
        self._best_network = self._network
        self._best_evaluator = self._evaluator
        self._best_trainer = self._trainer

    def update_best_model(self):
        assert self._trainer is not None

        if self._best_trainer is None:
            self._update_best_modules()

        if self._best_model_params is None:
            self._best_model_params = self._model_params

        if self._best_metrics_result is None:
            self._best_metrics_result = self._metrics_result

        if self._best_feature_list is None:
            self._best_feature_list = self._feature_list

        if self._metrics_result.result > self._best_metrics_result.result:
            self._update_best_modules()
            self._best_model_params = self._model_params
            self._best_metrics_result = self._metrics_result
            self._best_feature_list = self._feature_list

    def _parse_feature_config(self):
        cate_features, num_features = deque(), deque()
        feature_config = yaml_read(self._feature_config_root)

        for fea_name, info in feature_config.items():
            if info["used"]:

                if info["ftype"] != "numerical":
                    cate_features.append(fea_name)
                else:
                    num_features.append(fea_name)
        selected_feature = cate_features + num_features  
        self._categorical_features = cate_features
        self._numerical_features = num_features
        return selected_feature

    def parse_model_params(self):
        """temporarily for update model HPP """
        self._model_params = {}

    def train_init(self, **entity):
        assert(entity.has_key("train_dataset") and entity.has_key("val_dataset") and entity.has_key("metrics"))

        self.parse_model_params()

        # Phase 1. Load and transform Dataset -----------------------
        train_dataset = entity["train_dataset"]
        val_dataset = entity["val_dataset"]

        self._feature_list, self._feature_map = self._parse_feature_config()

        train_dataset.update_dataset_feature_config(self._feature_list)
        val_dataset.update_dataset_feature_config(self._feature_list)

        train_dataset.build()
        val_dataset.build()

        train_dataset.update_dataset_parameters(self._model_params["batch_size"]) 
        val_dataset.update_dataset_parameters(self._model_params["batch_size"])

        # Phase 2. Create Feature Statistics, and Run -----------------------
        self._statistics_gen = DatasetStatisticsGen(dataset=train_dataset,
                                                    categorical_features=self._categorical_features,
                                                    numerical_features=self._numerical_features)
        statistics =  self.statistics_gen.run()
        print(statistics)

        # Phase 3. Create Transform and Network, and Run -----------------------
        self._transform1 = CategoricalTransform(
            statistics=statistics,
            feature_names=self._categorical_features,
            embed_size=self._model_params["embed_size"],
            map_top_k_to_select=self._model_params["map_top_k_to_select"],
            map_shared_embedding=self._model_params["map_shared_embedding"],
        )
        self._transform2 = NumericalTransform(
            statistics=statistics,
            feature_names=self._numerical_features
        )

        Loss = LossFunctionFactory.get_loss_function(task_type=self._task_type)

        self._network = MlpNetwork(
            categorical_features=self._categorical_features,
            numerical_features=self._numerical_features,
            loss=Loss(label_name=train_dataset.target_name)
        )
        # Phase 4. Create Evaluator and Trainer
        self._metrics = { 
            entity["metrics"].name: entity["metrics"]
        }

        self._evaluator = Evaluator(
            dataset=val_dataset,
            transform_functions=[self._transform1.transform_fn, self._transform2.transform_fn],
            eval_fn=self._network.eval_fn,
            metrics=self._metrics,
        )

        self._trainer = Trainer(
            dataset=train_dataset,
            transform_functions=[self._transform1.transform_fn, self._transform2.transform_fn],
            train_fn=self._network.train_fn,
            validate_steps=self._model_params["validate_steps"],
            log_steps=self._model_params["log_steps"],
            learning_rate=self._model_params["learning_rate"],
            optimizer_type=self._model_params["optimizer_type"],
            train_epochs=self._model_params["train_epochs"],
            evaluator=self._evaluator,
            save_checkpoints_dir=self._save_checkpoints_dir,
            restore_checkpoint_path=self._restore_checkpoint_path,
            tensorboard_logdir=self._tensorboard_logdir
        )
        
    def inference_init(self, **entity):
        """load saved model for predict and eval."""
        pass

    def train(self, **entity):
        self.train_init(entity)
        self._trainer.run()
        self.update_best_model()
        
    def model_save(self):
        self._statistics.save_to_file(self._feature_statistics_filepath)
        saver = self._saver_init()
        saver.run()

    @abc.abstractmethod
    def predict(self, **entity):
        pass

    @abc.abstractmethod
    def eval(self, **entity):
        pass

    @abc.abstractmethod
    def get_train_metric(self):
        pass

    @abc.abstractmethod
    def get_train_loss(self):
        pass

    @abc.abstractmethod
    def get_val_loss(self):
        pass     

    @abc.abstractmethod
    def preprocess(self):
        """
        This method is used to implement Normalization, Standardization, Ont hot encoding which need
        self._train_flag parameters, and this operator needs model_config dict.
        :return: None
        """
        if self._train_flag:
            self._train_preprocess()
        else:
            self._predict_process()

    @abc.abstractmethod
    def _train_preprocess(self):
        pass

    @abc.abstractmethod
    def _predict_process(self):
        pass

    def _saver_init(self):
        if not self._trainer:
            saver = ModelSaver(
                transform_functions=[
                    self._select_transform,
                    self._transform1.transform_fn,
                    self._transform2.transform_fn
                ],
                serve_fn=self._network.serve_fn,
                restore_checkpoint_path=os.path.join(self._save_checkpoints_dir,
                                                    "ckpt_epoch-1"),
                save_model_dir=self._save_model_dir
            )
            return saver
        else:
            raise TypeError("trainer has not been initialized.")

    def _select_transform(self, example):
        features = tf.io.parse_example(example, self._feature_map)
        for k, v in features.items():
            features[k] = tf.identity(v)
        return features



