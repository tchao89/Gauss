# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic Inc. All rights reserved.
# Authors: Lab
"""

"""
from gauss_factory.abstarct_guass import AbstractGauss

from entity.dataset.plain_dataset import PlaintextDataset
from entity.dataset.multiprocess_plain_dataset import MultiprocessPlaintextDataset
from entity.feature_configuration.feature_config import FeatureConf
from entity.model.gbdt import GaussLightgbm
from entity.model.multiprocess_gbdt import MultiprocessGaussLightgbm
from entity.model.linear_models import GaussLinearModels
from entity.metrics.udf_metric import AUC
from entity.metrics.udf_metric import BinaryF1
from entity.metrics.udf_metric import MulticlassF1
from entity.metrics.udf_metric import MSE
from entity.losses.udf_loss import MSELoss
from entity.losses.udf_loss import BinaryLogLoss

"""
This class will be used in local_pipeline
"""
class EntityFactory(AbstractGauss):
    def get_entity(self, entity_name: str, **params):
        if entity_name is None:
            return None
        if entity_name.lower() == "plaindataset":
            # parameters: name: str, task_type: str, data_pair: Bunch, data_path: str, target_name: str, memory_only: bool
            return PlaintextDataset(**params)
        if entity_name.lower() == "multiprocess_plaindataset":
            # parameters: name: str, task_type: str, data_pair: Bunch, data_path: str, target_name: str, memory_only: bool
            return MultiprocessPlaintextDataset(**params)
        elif entity_name.lower() == "feature_configure":
            # parameters: name: str, file_path: str
            return FeatureConf(**params)
        elif entity_name.lower() == "auc":
            # parameters: name: str, label_name: str
            return AUC(**params)
        elif entity_name.lower() == "binary_f1":
            # parameters: name: str, label_name: str
            return BinaryF1(**params)
        elif entity_name.lower() == "multiclass_f1":
            return MulticlassF1(**params)
        elif entity_name.lower() == "mse":
            return MSE(**params)
        elif entity_name.lower() == "mse_loss":
            return MSELoss(**params)
        elif entity_name.lower() == "binary_logloss":
            return BinaryLogLoss(**params)
        elif entity_name.lower() == "lightgbm":
            # parameters: name: str, label_name: str
            return GaussLightgbm(**params)
        elif entity_name.lower() == "multiprocess_lightgbm":
            return MultiprocessGaussLightgbm(**params)
        elif entity_name.lower() == "lr":
            return GaussLinearModels(**params)
        elif entity_name.lower() == "multiprocess_lr":
            return GaussLinearModels(**params)

        raise ValueError("Entity factory can not construct entity by name: %s.", entity_name)

    def get_component(self, component_name: str):
        return None
