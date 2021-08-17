# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab
# 
import os
import copy

from entity.model.model import ModelWrapper
from core.tfdnn.trainers.trainer import Trainer
from core.tfdnn.evaluators.evaluator import Evaluator
from core.tfdnn.evaluators.predictor import Predictor
from core.tfdnn.networks.mlp_network import MlpNetwork
from core.tfdnn.model_savers.model_saver import ModelSaver
from entity.dataset.tf_plain_dataset import TFPlainDataset
from core.tfdnn.transforms.numerical_transform import NumericalTransform
from core.tfdnn.transforms.categorical_transform import CategoricalTransform
from core.tfdnn.statistics_gens.dataset_statistics_gen import DatasetStatisticsGen
from core.tfdnn.statistics_gens.external_statistics_gen import ExternalStatisticsGen
from utils.utils import tf_global_config
from utils.common_component import mkdir
from utils.common_component import feature_list_generator
from gauss_factory.loss_factory import LossFunctionFactory


class GaussNN(ModelWrapper):

    def __init__(self, **params):
        """"""
        super(GaussNN, self).__init__(
            name=params["name"],
            model_path=params["model_path"],
            model_config_root=params["model_config_root"],
            feature_config_root=params["feature_config_root"],
            task_type=params["task_type"],
            train_flag=params["train_flag"]
            )

        self._model_root = params["model_root"]
        self.model_file_name = params["model_root"] + "/" + self.name + ".txt"
        self.model_config_file_name = self._model_config_root + "/" + self.name + ".model_conf.yaml"
        self.feature_config_file_name = self._feature_config_root + "/" + self.name + ".final.yaml"

        self._save_statistics_filepath = self._model_root + "/statistics/"
        self._save_checkpoints_dir = self._model_root + "/checkpoint/"
        self._restore_checkpoint_dir = self._model_root + "/restore_checkpoint/"
        self._save_tensorboard_logdir = self._model_root + "/tensorboard_logdir/"
        self._save_model_dir = self._model_root + "/saved_model/"
        
        self._model_params = {}
        self._categorical_features = None
        self._best_categorical_features = None
        self._numerical_features = None
        self._best_numerical_features = None

        self._statistics_gen = None
        self._statistics = None
        self._transform1 = None
        self._transform2 = None
        self._network = None
        self._evaluator = None
        self._trainer = None

        self._create_folders()
        # tf_global_config(intra_threads=8, inter_threads=8)
        

    def __repr__(self):
        pass
    

    @property
    def val_metrics(self):
        return self._metrics.metrics_result
    
    
    def update_feature_conf(self, feature_conf):
        """select features and classify features to cate or num."""
        self._feature_conf = feature_conf
        self._feature_list = feature_list_generator(feature_conf=self._feature_conf)
        self._categorical_features, self._numerical_features = self._cate_num_split(configs=self._feature_conf)


    def train_init(self, **entity):
        import tensorflow as tf
        assert(entity.get("dataset") and entity.get("val_dataset") and entity.get("metrics"))
                
        tf.compat.v1.reset_default_graph()
        # Phase 1. Load and transform Dataset -----------------------
        train_dataset = self.preprocess(entity["dataset"])
        val_dataset = self.preprocess(entity["val_dataset"])

        train_dataset.update_features(self._feature_list, self._categorical_features)
        val_dataset.update_features(self._feature_list, self._categorical_features)

        train_dataset.build()
        val_dataset.build()

        train_dataset.update_dataset_parameters(self._model_params["batch_size"]) 
        val_dataset.update_dataset_parameters(self._model_params["batch_size"])

        # Phase 2. Create Feature Statistics, and Run -----------------------
        statistics_gen = DatasetStatisticsGen(
            dataset=train_dataset,
            categorical_features=self._categorical_features,
            numerical_features=self._numerical_features
        )
        self._statistics = statistics_gen.run()

        # Phase 3. Create Transform and Network, and Run -----------------------
        self._transform1 = CategoricalTransform(
            statistics=self._statistics,
            feature_names=self._categorical_features,
            embed_size=self._model_params["embed_size"],
        )
        self._transform2 = NumericalTransform(
            statistics=self._statistics,
            feature_names=self._numerical_features
        )
        Loss = LossFunctionFactory.get_loss_function(task_type=self._task_type)
        self._network = MlpNetwork(
            categorical_features=self._categorical_features,
            numerical_features=self._numerical_features,
            hidden_sizes=self._model_params["hidden_sizes"],
            loss=Loss(label_name=train_dataset.target_name)
        )
        # Phase 4. Create Evaluator and Trainer
        self._metrics = entity["metrics"] 
        metrics_wrapper = { 
            self._metrics.name: self._metrics
        }
        self._evaluator = Evaluator(
            dataset=val_dataset,
            transform_functions=[self._transform1.transform_fn, self._transform2.transform_fn],
            eval_fn=self._network.eval_fn,
            metrics=metrics_wrapper,
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
            tensorboard_logdir=self._save_tensorboard_logdir
        )
        
    def inference_init(self, **entity):
        """load saved model for predict and eval."""
        assert(entity.get("val_dataset"))

        dataset = self.preprocess(entity["val_dataset"])
        dataset.update_features(self._feature_list, self._categorical_features)
        dataset.build()

        statistic_gen = ExternalStatisticsGen(
            filepath=self._save_statistics_filepath + "statistics.pkl"
        )
        statistics = statistic_gen.run()

        self._transform1 = CategoricalTransform(
            statistics=statistics,
            feature_names=self._categorical_features,
            embed_size=self._model_params["embed_size"],
        )
        self._transform2 = NumericalTransform(
            statistics=statistics,
            feature_names=self._numerical_features
        )
        self._network = MlpNetwork(
            categorical_features=self._categorical_features,
            numerical_features=self._numerical_features,
            hidden_sizes=self._model_params["hidden_sizes"],
        )
        if self._train_flag:
            self._metrics = entity["metrics"]
            metrics_wrapper = { 
            self._metrics.name: self._metrics
            }
            self._evaluator = Evaluator(
                dataset=dataset,
                transform_functions=[self._transform1.transform_fn, self._transform2.transform_fn],
                eval_fn=self._network.eval_fn,
                metrics=metrics_wrapper,
                restore_checkpoint_dir=self._restore_checkpoint_dir,
            )
        else:
            self._evaluator = Predictor(
                dataset=dataset,
                transform_functions=[self._transform1.transform_fn, self._transform2.transform_fn],
                eval_fn=self._network.eval_fn,
                restore_checkpoint_dir=self._restore_checkpoint_dir,
            )

    def train(self, **entity):
        assert self._train_flag
        self.train_init(**entity)
        self._trainer.run()

    def predict(self, **entity):
        self.inference_init(**entity)
        predict = self._inference_evaluator.run()
        return predict

    def eval(self, **entity):
        if self._train_flag:
            print(self._metrics.metrics_result)
            return self._metrics.metrics_result
        else:
            self.inference_init(**entity)
            self._inference_evaluator.run()

    def update_best_model(self):
        assert self._trainer is not None

        if self._best_model_params is None or \
            (self._metrics.metrics_result.result > self._best_metrics_result):

            self._update_checkpoint()
            self._update_statistics()
            self._best_model_params = copy.deepcopy(self._model_params)
            self._best_metrics_result = self._metrics.metrics_result.result
            self._best_feature_list = copy.deepcopy(self._feature_list)
            self._best_categorical_features = copy.deepcopy(self._categorical_features)
            self._best_numerical_features = copy.deepcopy(self._numerical_features)

    def set_best_model(self):
        self._model_params = copy.deepcopy(self._best_model_params)
        self._feature_list = copy.deepcopy(self._best_feature_list)
        self._categorical_features = copy.deepcopy(self._best_categorical_features)
        self._numerical_features = copy.deepcopy(self._best_numerical_features)

    def get_train_metric(self):
        pass

    def get_train_loss(self):
        pass

    def get_val_loss(self):
        pass     

    def preprocess(self, dataset):
        """
        This method is used to implement Normalization, Standardization, Ont hot encoding which need
        self._train_flag parameters, and this operator needs model_config dict.
        :return: None
        """
        dataset = TFPlainDataset(
            name="tf_dataset",
            dataset=dataset,
            task_type=dataset.task_type,
            target_name=dataset.target_name
        )
        return dataset

    def _train_preprocess(self):
        pass

    def _predict_process(self):
        pass

    def set_best(self):
        pass

    def update_best(self):
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
    
    def model_save(self):
        self._statistics.save_to_file(self._feature_statistics_filepath)
        saver = self._saver_init()
        saver.run()
    
    # def _select_transform(self, example):
    #     features = tf.io.parse_example(example, self._feature_map)
    #     for k, v in features.items():
    #         features[k] = tf.identity(v)
    #     return features

    def _create_folders(self):
        if not os.path.isdir(self._save_statistics_filepath):
            mkdir(self.__save_statistics_filepath)
        if not os.path.isdir(self._save_checkpoints_dir):
            mkdir(self._save_checkpoints_dir)
        if not os.path.isdir(self._restore_checkpoint_dir):
            mkdir(self._restore_checkpoint_dir)
        if not os.path.isdir(self._save_tensorboard_logdir):
            mkdir(self._save_tensorboard_logdir)
        if not os.path.isdir(self._save_model_dir):
            mkdir(self._save_model_dir)

    def _cate_num_split(self, configs):
        configs = configs.feature_dict
        categorical_features, numerical_features = [], []
        for fea, info in configs.items():
            if info["ftype"] != "numerical":
                categorical_features.append(fea)
            else:
                numerical_features.append(fea)
        return categorical_features, numerical_features

    def _update_checkpoint(self):
        import shutil
        import tensorflow as tf

        if os.path.isdir(self._restore_checkpoint_dir):
            shutil.rmtree(self._restore_checkpoint_dir)
        os.mkdir(self._restore_checkpoint_dir)
        prefix = tf.train.latest_checkpoint(self._save_checkpoints_dir) + "*"
        os.system("cp {ckpt_dir} {target_dir}".format(
            ckpt_dir=self._save_checkpoints_dir + "checkpoint",
            target_dir=self._restore_checkpoint_dir
        ))
        os.system("cp {ckpt} {target_dir}".format(
            ckpt=prefix,
            target_dir=self._restore_checkpoint_dir
        ))

    def _update_statistics(self):
        self._statistics.save_to_file(
            filepath=self._save_statistics_filepath+"statistics.pkl"
        )

    # def _parse_feature_config(self):
        # cate_features, num_features = deque(), deque()
        # feature_config = yaml_read(self._feature_config_root)

        # for fea_name, info in feature_config.items():
        #     if info["used"]:

        #         if info["ftype"] != "numerical":
        #             cate_features.append(fea_name)
        #         else:
        #             num_features.append(fea_name)
        # selected_feature = cate_features + num_features  
        # self._categorical_features = cate_features
        # self._numerical_features = num_features
        # return selected_feature
