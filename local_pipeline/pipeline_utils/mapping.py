# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
"""
This is a root path configure file for feature
"""
import os
import random
import string

from utils.bunch import Bunch


class EnvironmentConfigure:
    """
    EnvironmentConfigure object
    """
    def __init__(self, work_root=None, user_feature=None):
        self.work_root = os.path.join(work_root, self.id_generator())

        self.user_feature_name = None
        self.user_feature_root = None
        self.user_feature_path = None

        if user_feature is not None:
            user_feature_root, user_feature_file = os.path.split(user_feature)
            self.user_feature_name = user_feature_file
            self.user_feature_root = user_feature_root
            self.user_feature_path = user_feature

    @classmethod
    def id_generator(cls,
                     size=6,
                     chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
        """
        Generate experiment id
        :param size:
        :param chars:
        :return:
        """
        return ''.join(random.choice(chars) for _ in range(size))

    @staticmethod
    def feature_dict():
        """
        Configuration files name
        :return:
        """
        feature_dict = Bunch()

        feature_dict.type_inference_feature = "type_inference_feature.yaml"
        feature_dict.data_clear_feature = "data_clear_feature.yaml"
        feature_dict.feature_generator_feature = "feature_generator_feature.yaml"
        feature_dict.unsupervised_feature = "unsupervised_feature.yaml"
        feature_dict.supervised_feature = "supervised_feature.yaml"
        feature_dict.label_encoding_path = "label_encoding_models"
        feature_dict.impute_path = "impute_models"
        feature_dict.final_feature_configure = "final_feature_configure.yaml"
        feature_dict.label_encoder_feature = "label_encoder_feature.yaml"
        feature_dict.train_user_configure = "train_user_config.yaml"
        feature_dict.pipeline_configure = "pipeline_configure.yaml"
        feature_dict.inference_configure = "inference_configure.yaml"

        return feature_dict
