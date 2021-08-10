import os
import sys
from numpy import float32, int64
import yaml
import warnings
import tensorflow as tf
BASE_DIR = '/home/gzqq/developer/CITIC_PLATFORM/Gauss_nn/'
sys.path.append(BASE_DIR)

from pprint import pprint
from icecream import ic

from entity.dataset.plain_dataset import PlaintextDataset
from entity.dataset.tf_plain_dataset import TFPlainDataset
from core.tfdnn.statistics_gens.dataset_statistics_gen import DatasetStatisticsGen
from core.tfdnn.transforms.categorical_transform import CategoricalTransform
from core.tfdnn.transforms.numerical_transform import NumericalTransform
from core.tfdnn.networks.mlp_network import MlpNetwork
from core.tfdnn.evaluators.evaluator import Evaluator
from core.tfdnn.model_savers.model_saver import ModelSaver
from core.tfdnn.trainers.trainer import Trainer
from entity.metrics.udf_metric import NNAUC
from gauss_factory.loss_factory import LossFunctionFactory

class CONFIGS:
    file_path = "./test_dataset/bank_gened.csv"
    target_name = ["deposit"]
    feature_config = "./test_dataset/unsupervised_feature.yaml"
    task_type = "classification"

    selected_features = ["age", "loan", "balance", "marital", "job", "education", 
    "housing", "month", "day", "PERCENTILE(balance)", "PERCENTILE(duration)"]
    categorical_features = ["age", "loan", "marital", "job", "education", "housing", "month",
        "day"]
    numerical_features = ["balance", "PERCENTILE(balance)", "PERCENTILE(duration)"]
    embed_size = 32

    # trainer param
    validate_steps = 300
    log_steps = 300
    learning_rate = 1e-3
    optimizer_type = "adam"
    train_epochs = 30

    save_checkpoints_dir = "./test_dataset/ckpt/"    
    restore_checkpoint_path = None
    save_model_dir  = "./test_dataset/save_model/"

    # feature_map = {
    #     "age": tf.io.FixedLenFeature(1, int64),
    #     "loan": tf.io.FixedLenFeature(1, int64),
    #     "balance": tf.io.FixedLenFeature(1, float32),
    #     "deposit": tf.io.FixedLenFeature(1, int64)
    # }

dataset_train = PlaintextDataset(name="test",
                                data_path=CONFIGS.file_path,
                                task_type=CONFIGS.task_type,
                                target_name=CONFIGS.target_name,
                                memory_only=True
                                )
dataset_val = dataset_train.split()

# print(dataset_train.get_dataset().data)
# print(dataset_val.get_dataset().data)

dataset_train = TFPlainDataset(name="tf_test",
                                dataset=dataset_train,
                                feature_config=CONFIGS.feature_config,
                                task_type=CONFIGS.task_type,
                                target_name=CONFIGS.target_name,
                                memory_only=True)
dataset_val = TFPlainDataset(name="valset",
                                dataset=dataset_val,
                                feature_config=CONFIGS.feature_config,
                                task_type=CONFIGS.task_type,
                                target_name=CONFIGS.target_name,
                                memory_only=True)

dataset_train.update_dataset_feature_config(CONFIGS.selected_features)
dataset_val.update_dataset_feature_config(CONFIGS.selected_features)

dataset_train.build()
dataset_val.build()


statistic_gen = DatasetStatisticsGen(dataset=dataset_train,
                                    categorical_features=CONFIGS.categorical_features,
                                    numerical_features=CONFIGS.numerical_features)
statistics = statistic_gen.run()
print(statistics)

transform1 = CategoricalTransform(
    statistics=statistics,
    feature_names=CONFIGS.categorical_features,
    # map_top_k_to_select=CONFIGS.map_top_k_to_select,
    embed_size=CONFIGS.embed_size
)
transform2 = NumericalTransform(
    statistics=statistics,
    feature_names=CONFIGS.numerical_features
)

Loss = LossFunctionFactory.get_loss_function(CONFIGS.task_type)
network = MlpNetwork(
    categorical_features=CONFIGS.categorical_features,
    numerical_features=CONFIGS.numerical_features,
    loss=Loss(label_name=CONFIGS.target_name),
)
metrics = {
    'auc': NNAUC(name="auc",label_name=CONFIGS.target_name),
}
evaluator = Evaluator(
    dataset=dataset_val,
    transform_functions=[transform1.transform_fn, transform2.transform_fn],
    eval_fn=network.eval_fn,
    metrics=metrics,
)

trainer = Trainer(
    dataset=dataset_train,
    transform_functions=[transform1.transform_fn, transform2.transform_fn],
    train_fn=network.train_fn,
    validate_steps=CONFIGS.validate_steps,
    log_steps=CONFIGS.log_steps,
    learning_rate=CONFIGS.learning_rate,
    optimizer_type=CONFIGS.optimizer_type,
    train_epochs=CONFIGS.train_epochs,
    evaluator=evaluator,
    save_checkpoints_dir=CONFIGS.save_checkpoints_dir,
    restore_checkpoint_path=CONFIGS.restore_checkpoint_path,
    # tensorboard_logdir=self._tensorboard_logdir
)

trainer.run()

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
#                     restore_checkpoint_path=os.path.join(CONFIGS.save_checkpoints_dir, "ckpt_epoch-1"), 
#                     save_model_dir=CONFIGS.save_model_dir)

# saver.run()
# ic(metrics["auc"])
# print(metrics)

# with tf.Session() as sess:
#     dataset_train.init(sess)
#     print(dataset_train.next_batch)
#     print(tf_dataset_test.numerical_features)