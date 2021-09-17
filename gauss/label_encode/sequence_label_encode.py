# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab
import shelve

from sklearn.preprocessing import LabelEncoder

from gauss.label_encode.base_label_encode import BaseLabelEncode
from utils.base import reduce_data
from utils.common_component import (
    yaml_read, 
    yaml_write
)


class SequenceLabelEncode(BaseLabelEncode):
    """Hash the categorical feature values for sequence dataset.

    Features has `ftype` is category or bool will normalized to 
    numbers, which stands for original string or bool values. In 
    training process, encoder will be saved as serialized binary
    file after fit and transform, when in predicting, binary encoder
    file will be loaded to raplace categorical values in infer 
    dataset by transfed values. 

    Parameters:
    ----------
    feature_configure_path : Path of configuration yaml file to
        be loaded.

    save_config_path : Path of generated configuration yaml to 
        be saved.

    save_encoder_path : Path of fitted encoder to be saved.
    
    Examples:
    ----------
    >>> encoder = SequenceLabelEncode(...)
    >>> encoder.run(dataset=dataset)
    """

    CATE = "category"
    BOOL = "bool"

    CLS = "classification"

    def __init__(self, **params):
        """
        Parameters:
        ----------
        feature_configure_path : Path of configuration yaml file to
            be loaded.

        save_config_path : Path of generated configuration yaml to 
            be saved.

        save_encoder_path : Path of fitted encoder to be saved
        """
          
        super(SequenceLabelEncode, self).__init__(
            name=params["name"],
            train_flag=params["train_flag"]\
                if params.get("train_flag") else True,
            enable=params["enable"]\
                if params.get("enable") else True,
            task_name=params["task_name"],
            feature_configure_path=params["feature_config_path"]
        )
        self._save_config_path = params["save_config_path"]
        self._save_encoder_path = params["save_encoder_path"]

        self._feature_config = None
        self._encoders = {}

    
    def _train_run(self, **entity):
        dataset = entity["train_dataset"]
        self._load_feature_config()
        self._encode(dataset=dataset)
        self._save_serialized()
        self._final_configure_generation()

    def _predict_run(self, **entity):
        dataset = entity["infer_dataset"]
        data = dataset.get_dataset().data
        label = dataset.get_dataset().target
        feature_names = dataset.get_dataset().feature_names
        label_name = dataset.get_dataset().label_names

        with shelve.open(self._save_encoder_path) as file:
            encoders = file["encoders"]
            self._feature_encode(data, feature_names, encoders)
            self._label_encode(label, label_name, encoders)

    def _encode(self, dataset):
        data = dataset.get_dataset().data
        feature_names = dataset.get_dataset().feature_names
        label = dataset.get_dataset().target
        label_name = dataset.get_dataset().target_names

        self._feature_encode(data, feature_names)
        self._label_encode(label, label_name)

        reduce_data(dataframe=data)
    
    def _feature_encode(self, data, feature_names, encoders=None):
        if self._train_flag:
            for fea_name in feature_names:
                if self._feature_config[fea_name]["ftype"] == self.CATE or \
                    self._feature_config[fea_name]["ftype"] == self.BOOL:
                    encoder = LabelEncoder().fit(data.loc[:, fea_name])
                    data[fea_name] = encoder.transform(data.loc[:, fea_name])
                    self._encoders[fea_name] = encoder
        else:
            if encoders is None:
                raise AttributeError(
                    "encoders must provide when predicting."
                    "Please check the `train_flag` or encoders."
                    )
            for fea_name in feature_names:
                if self._feature_config[fea_name]["ftype"] == self.CATE or\
                    self._feature_config[fea_name]["ftype"] == self.BOOL:
                    encoder = encoders[fea_name]
                    # hash_table = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                    # blank for handle oov features.
                    data[fea_name] = encoder.transform(data[fea_name])

    def _label_encode(self, label, label_name: str, encoders=None):
        if self._train_flag:
            if self._task_name == self.CLS:
                encoder = LabelEncoder().fit(label)
                label = encoder.transform(label)
                self._encoders[label_name] = encoder
        else:
            if encoders is None:
                raise AttributeError(
                    "encoders must provide when predicting."
                    "Please check the `train_flag` or encoders."
                    )
            if self._task_name == self.CLS:
                encoder = encoders[label_name]
                # blank for handle OOV.
                label = encoder.transform(label)


    def _final_configure_generation(self):
        yaml_write(yaml_dict=self._feature_config, yaml_file=self._save_config_path)

    def _load_feature_config(self):
        self._feature_config = yaml_read(self._feature_configure_path)

    def _save_serialized(self):
        with shelve.open(self._save_encoder_path) as file:
            file["encoders"] = self._encoders