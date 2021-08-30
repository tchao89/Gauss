import os
import sys
from numpy import float32, int64
import yaml
import warnings
import tensorflow as tf
BASE_DIR = '/home/gzqq/developer/CITIC_PLATFORM/Gauss_nn/'
sys.path.append(BASE_DIR)

from icecream import ic

from entity.dataset.plain_dataset import PlaintextDataset
from entity.dataset.tf_plain_dataset import TFPlainDataset
from core.tfdnn.statistics_gens.dataset_statistics_gen import DatasetStatisticsGen
from core.tfdnn.statistics_gens.external_statistics_gen import ExternalStatisticsGen
from core.tfdnn.transforms.categorical_transform import CategoricalTransform
from core.tfdnn.transforms.numerical_transform import NumericalTransform
from core.tfdnn.networks.mlp_network import MlpNetwork
from core.tfdnn.evaluators.evaluator import Evaluator
from core.tfdnn.evaluators.predictor import Predictor
from core.tfdnn.trainers.trainer import Trainer
from entity.metrics.udf_metric import NNAUC
from gauss_factory.loss_factory import LossFunctionFactory

class CONFIGS:
    train_set = "./test_dataset/bank_numerical_train_realdata.csv"
    val_set = "./test_dataset/bank_numerical_val_realdata.csv"
    target_name = ["deposit"]
    feature_config = "./test_dataset/unsupervised_feature.yaml"
    task_type = "classification"

    selected_features = ["age", "loan", "balance", "marital", "job", "education", "housing", "duration", 
                         "day", "month"]
    categorical_features = ["age", "loan", "marital", "job", "education", "housing", "day", "month"]
    numerical_features = ["balance", "duration"]
    embed_size = 8

    # trainer param
    validate_steps = 300
    log_steps = 300
    learning_rate = 1e-3
    optimizer_type = "adam"
    train_epochs = 5

    save_statistics_path = "./test_dataset/stat/stat.pkl"
    save_checkpoints_dir = "./test_dataset/ckpt/"    
    restore_checkpoint_path = None
    save_model_dir  = "./test_dataset/save_model/"

    test_set = "./test_dataset/bank_numerical_test_realdata.csv"

    feature_map = {
        "age": tf.io.FixedLenFeature(1, tf.int64),
        "loan": tf.io.FixedLenFeature(1, tf.int64),
        "balance": tf.io.FixedLenFeature(1, tf.float32),
        "marital": tf.io.FixedLenFeature(1, tf.int64),
        "job": tf.io.FixedLenFeature(1, tf.int64),
        "education": tf.io.FixedLenFeature(1, tf.int64),
        "housing": tf.io.FixedLenFeature(1, tf.int64),
        "duration": tf.io.FixedLenFeature(1, tf.float32),
        "day": tf.io.FixedLenFeature(1, tf.int64),
        "month": tf.io.FixedLenFeature(1, tf.int64),
        "deposit": tf.io.FixedLenFeature(1, int64)
    }

dataset_train = PlaintextDataset(name="train_set",
                                data_path=CONFIGS.train_set,
                                task_type=CONFIGS.task_type,
                                target_name=CONFIGS.target_name,
                                memory_only=True
                                )
dataset_val = PlaintextDataset(name="val_set",
                                data_path=CONFIGS.val_set,
                                task_type=CONFIGS.task_type,
                                target_name=CONFIGS.target_name,
                                memory_only=True
                                )

print(dataset_train.get_dataset().data)
print(dataset_val.get_dataset().data)

dataset_train = TFPlainDataset(name="tf_train_set",
                                dataset=dataset_train,
                                task_type=CONFIGS.task_type,
                                target_name=CONFIGS.target_name,
                                memory_only=True)

dataset_val = TFPlainDataset(name="tf_val_set",
                                dataset=dataset_val,
                                task_type=CONFIGS.task_type,
                                target_name=CONFIGS.target_name,
                                memory_only=True)

# dataset_train.update_features(CONFIGS.selected_features, CONFIGS.categorical_features)
# dataset_val.update_features(CONFIGS.selected_features, CONFIGS.categorical_features)

# dataset_train.build()
# dataset_val.build()

# statistic_gen = DatasetStatisticsGen(dataset=dataset_train,
#                                     categorical_features=CONFIGS.categorical_features,
#                                     numerical_features=CONFIGS.numerical_features)
# statistics = statistic_gen.run()
# statistics.save_to_file(CONFIGS.save_statistics_path)
# print(statistics)

# transform1 = CategoricalTransform(
#     statistics=statistics,
#     feature_names=CONFIGS.categorical_features,
#     embed_size=CONFIGS.embed_size
# )
# transform2 = NumericalTransform(
#     statistics=statistics,
#     feature_names=CONFIGS.numerical_features
# )

# Loss = LossFunctionFactory.get_loss_function(CONFIGS.task_type)
# network = MlpNetwork(
#     categorical_features=CONFIGS.categorical_features,
#     numerical_features=CONFIGS.numerical_features,
#     loss=Loss(label_name=CONFIGS.target_name),
# )
# metrics = {
#     'auc': NNAUC(name="auc",label_name=CONFIGS.target_name),
# }
# evaluator = Evaluator(
#     dataset=dataset_val,
#     transform_functions=[transform1.transform_fn, transform2.transform_fn],
#     eval_fn=network.eval_fn,
#     metrics=metrics,
# )

# trainer = Trainer(
#     dataset=dataset_train,
#     transform_functions=[transform1.transform_fn, transform2.transform_fn],
#     train_fn=network.train_fn,
#     validate_steps=CONFIGS.validate_steps,
#     log_steps=CONFIGS.log_steps,
#     learning_rate=CONFIGS.learning_rate,
#     optimizer_type=CONFIGS.optimizer_type,
#     train_epochs=CONFIGS.train_epochs,
#     evaluator=evaluator,
#     save_checkpoints_dir=CONFIGS.save_checkpoints_dir,
#     restore_checkpoint_path=CONFIGS.restore_checkpoint_path,
#     # tensorboard_logdir=self._tensorboard_logdir
# )

# trainer.run()
# print("train res: ", metrics["auc"])

# def select_transform(example):
#     features = tf.io.parse_example(example, CONFIGS.feature_map)
#     for k, v in features.items():
#         features[k] = tf.identity(v)
#     return features

# saver = ModelSaver(transform_functions=[
#                             select_transform, 
#                             transform1.transform_fn, 
#                             transform2.transform_fn],
#                     serve_fn=network.serve_fn,
#                     restore_checkpoint_path=CONFIGS.save_checkpoints_dir, 
#                     save_model_dir=CONFIGS.save_model_dir)

# saver.run()

# dataset_test = PlaintextDataset(name="test",
#                             data_path=CONFIGS.val_set,
#                             task_type=CONFIGS.task_type,
#                             memory_only=True
#                             )

# dataset_test = TFPlainDataset(name="tf_test_set",
#                             dataset=dataset_test,
#                             target_name=CONFIGS.target_name,
#                             task_type=CONFIGS.task_type,
#                             )
# dataset_test.update_features(CONFIGS.selected_features, CONFIGS.categorical_features)
# dataset_test.build()

# test_metrics = {
#     'auc': NNAUC(name="auc",label_name=CONFIGS.target_name),
# }

# test_evaluator = Evaluator(
#     dataset=dataset_test,
#     transform_functions=[transform1.transform_fn, transform2.transform_fn],
#     eval_fn=network.eval_fn,
#     metrics=test_metrics,
#     restore_checkpoint_path=CONFIGS.save_checkpoints_dir
# )
# test_evaluator.run()
# print("evaluate res: ", test_metrics["auc"])

# dataset_test = PlaintextDataset(name="test",
#                             data_path=CONFIGS.test_set,
#                             task_type=CONFIGS.task_type,
#                             memory_only=True
#                             )

# dataset_test = TFPlainDataset(name="tf_test_set",
#                             dataset=dataset_test,
#                             task_type=CONFIGS.task_type,
#                             )

# dataset_test.update_features(CONFIGS.selected_features, CONFIGS.categorical_features)
# dataset_test.build()

# statistic_gen = ExternalStatisticsGen(filepath=CONFIGS.save_statistics_path)
# statistics = statistic_gen.run()

# transform1 = CategoricalTransform(
#     statistics=statistics,
#     feature_names=CONFIGS.categorical_features,
#     embed_size=CONFIGS.embed_size
# )
# transform2 = NumericalTransform(
#     statistics=statistics,
#     feature_names=CONFIGS.numerical_features
# )

# network = MlpNetwork(
#     categorical_features=CONFIGS.categorical_features,
#     numerical_features=CONFIGS.numerical_features,
# )

# test_predictor = Predictor(dataset=dataset_test,
#                             transform_functions=[transform1.transform_fn, transform2.transform_fn],
#                             eval_fn=network.eval_fn,
#                             restore_checkpoint_path=CONFIGS.save_checkpoints_dir)
# res = test_predictor.run()
# ic(res)
# with tf.Session() as sess:
#     dataset_train.init(sess)
#     print(dataset_train.next_batch)
#     print(tf_dataset_test.numerical_features)