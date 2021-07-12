# This is a root path configure file for feature
import os
import datetime
import random
import string

from utils.bunch import Bunch


class EnvironmentConfigure(object):
    def __init__(self, env):
        assert env in ["prod", "test"]
        self._env = env
        self.work_root = None
        self.type_inference_feature_path = None
        self.feature_generate_path = None
        self.unsupervised_feature_selector_path = None
        self.supervise_feature_selector_path = None

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env: str):
        self._env = env

    def env_conf(self,
                 root="/home/liangqian/PycharmProjects/Gauss/experiments",
                 user_feature="/home/liangqian/PycharmProjects/Gauss/test_dataset/feature_conf.yaml"):

        assert self._env in ["prod", "test"]

        if self.env == "prod":
            self.type_inference_feature_path = "type-inference.feature"
            self.feature_generate_path = "fea-gen.feature"
            self.unsupervised_feature_selector_path = "unsupervised-selector.feature"
            self.supervise_feature_selector_path = "supervised-selector.feature"

        else:
            time = datetime.datetime.now()
            experiment_id = datetime.datetime.strftime(time, '%Y%m%d-%H:%M--') + self.id_generator()
            experiment_path = os.path.join(root, experiment_id)
            os.mkdir(experiment_path)
            model_path = os.path.join(experiment_path, "model_path")
            os.mkdir(model_path)

            feature_dict = Bunch()
            feature_dict.user_feature = user_feature
            feature_dict.type_inference_feature = os.path.join(experiment_path, "type_inference_feature.yaml")
            feature_dict.data_clear_feature = os.path.join(experiment_path, "data_clear_feature.yaml")
            feature_dict.feature_generator_feature = os.path.join(experiment_path, "feature_generator_feature.yaml")
            feature_dict.unsupervised_feature = os.path.join(experiment_path, "unsupervised_feature.yaml")
            feature_dict.label_encoding_path = os.path.join(experiment_path, "label_encoding_models")

            return experiment_path, feature_dict

    @classmethod
    def id_generator(cls, size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
        return ''.join(random.choice(chars) for _ in range(size))
