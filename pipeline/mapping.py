# This is a root path configure file for feature
import os
import random
import string

from utils.bunch import Bunch


class EnvironmentConfigure(object):
    def __init__(self, work_root, user_feature):
        self.work_root = os.path.join(work_root, self.id_generator())
        user_feature_root, user_feature_file = os.path.split(user_feature)

        self.feature_path = "/feature"
        self.pipeline_path = "/pipeline"
        self.model_path = "/model"

        self.user_feature_name = user_feature_file
        self.user_feature_root = user_feature_root
        self.user_feature_path = user_feature
        self.type_inference_feature_path = None
        self.type_inference_feature = "type_inference_feature.yaml"
        self.data_clear_feature_path = None
        self.data_clear_feature = "data_clear_feature.yaml"
        self.feature_generate_path = None
        self.feature_generator_feature = "feature_generate_feature.yaml"
        self.label_encoding_path = None
        self.label_encoding_models = "label_encoding_models"
        self.unsupervised_feature_selector_path = None
        self.unsupervised_feature = "unsupervised_feature.yaml"
        self.supervised_feature_selector_path = None
        self.supervised_feature = "supervised_selector.yaml"

        self.model_save_path = None
        self.model_config_path = None
        self.model_config = "model_config.yaml"

        self.create_path()

    def create_path(self):
        self.env_conf()

    @property
    def file_name(self):
        config_path_dict = Bunch()

        config_path_dict.user_feature_name = self.user_feature_name
        config_path_dict.type_inference_feature = self.type_inference_feature
        config_path_dict.data_clear_feature = self.data_clear_feature
        config_path_dict.feature_generator_feature = self.feature_generator_feature
        config_path_dict.label_encoding_models = self.label_encoding_models
        config_path_dict.unsupervised_feature = self.unsupervised_feature
        config_path_dict.supervised_feature = self.supervised_feature
        config_path_dict.model_config = self.model_config

        return config_path_dict

    @property
    def file_path(self):
        config_path_dict = Bunch()

        config_path_dict.user_feature = self.user_feature_path
        config_path_dict.type_inference_feature = self.type_inference_feature_path + "/" + self.type_inference_feature
        config_path_dict.data_clear_feature = self.data_clear_feature_path + "/" + self.data_clear_feature
        config_path_dict.feature_generator_feature = self.feature_generate_path + "/" + self.feature_generator_feature
        config_path_dict.label_encoding_path = self.label_encoding_path + "/" + self.label_encoding_models
        config_path_dict.unsupervised_feature = self.unsupervised_feature_selector_path + "/" + self.unsupervised_feature
        config_path_dict.supervised_feature = self.supervised_feature_selector_path + "/" + self.supervised_feature
        config_path_dict.model_config = self.model_config_path + "/" + self.model_config

        return config_path_dict

    @property
    def root_name(self):
        config_path_dict = Bunch()

        config_path_dict.user_feature_root = self.user_feature_root
        config_path_dict.type_inference_feature_path = self.type_inference_feature_path
        config_path_dict.data_clear_feature_path = self.data_clear_feature_path
        config_path_dict.feature_generate_path = self.feature_generate_path
        config_path_dict.label_encoding_path = self.label_encoding_path
        config_path_dict.unsupervised_feature_selector_path = self.unsupervised_feature_selector_path
        config_path_dict.supervised_feature_selector_path = self.supervised_feature_selector_path
        config_path_dict.model_save_path = self.model_save_path
        config_path_dict.model_config_path = self.model_config_path

        return config_path_dict

    def env_conf(self):
        self.feature_path = self.work_root + self.feature_path
        self.pipeline_path = self.work_root + self.pipeline_path
        self.model_path = self.work_root + self.model_path

        self.type_inference_feature_path = self.feature_path + "/type_inference"
        self.data_clear_feature_path = self.feature_path + "/data_clear"
        self.feature_generate_path = self.feature_path + "/data_generate_feature"
        self.label_encoding_path = self.feature_path + "/label_encoding_path"
        self.supervised_feature_selector_path = self.feature_path + "/supervised_selector"
        self.unsupervised_feature_selector_path = self.feature_path + "/unsupervised_selector"

        self.model_save_path = self.model_path + "/model_save/"
        self.model_config_path = self.model_path + "/model_config/"

    @classmethod
    def id_generator(cls, size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
        return ''.join(random.choice(chars) for _ in range(size))
