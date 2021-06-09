# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import shelve

import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

import featuretools as ft
from entity.base_dataset import BaseDataset
from gauss.feature_generation.base_feature_generation import BaseFeatureGenerator
from utils.Logger import logger


class FeatureToolsGenerator(BaseFeatureGenerator):

    def __init__(self, name, train_flag, enable, feature_config_path, label_encoding_configure_path):
        super(FeatureToolsGenerator, self).__init__(name=name,
                                                    train_flag=train_flag,
                                                    enable=enable,
                                                    feature_configure_path=feature_config_path)
        self.entity_set = ft.EntitySet(id=self.name)
        self.label_encoding_configure_path = label_encoding_configure_path
        self.label_encoding = {}

        self.variable_types = {}
        self.feature_configure = None

    def _train_run(self, **entity):

        assert "dataset" in entity.keys()
        dataset = entity["dataset"]

        self._set_feature_configure()
        assert self.feature_configure is not None

        self._label_encoding(dataset=dataset)
        self._label_encoding_serialize()
        self._ft_generator(dataset=dataset)

    def _predict_run(self, **entity):

        assert "dataset" in entity.keys()
        dataset = entity['dataset']

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)

        feature_names = dataset.get_dataset().feature_names

        self._set_feature_configure()
        assert self.feature_configure is not None

        with shelve.open(self.label_encoding_configure_path) as shelve_open:
            le_model_list = shelve_open['label_encoding']

            for col in feature_names:
                if self.feature_configure[col]['ftype'] == "category":
                    assert le_model_list.get(col)
                    le_model = le_model_list[col]

                    label_dict = dict(zip(le_model.classes_, le_model.transform(le_model.classes_)))
                    status_list = data[col].unique().tolist()

                    for item in status_list:
                        if label_dict.get(item) is None:
                            logger.info("feature: " + str(col) + "has an abnormal value (unseen by label encoding): " + str(item))
                            raise ValueError("feature: " + str(col) + " has an abnormal value (unseen by label encoding): " + str(item))

                    data[col] = le_model.transform(data[col])

        self._ft_generator(dataset=dataset)

    def _set_feature_configure(self):

        feature_configure_file = open(self.feature_configure_path, 'r', encoding='utf-8')
        feature_configure = feature_configure_file.read()
        self.feature_configure = yaml.load(feature_configure, Loader=yaml.FullLoader)
        feature_configure_file.close()

    def _ft_generator(self, dataset: BaseDataset):
        data = dataset.get_dataset().data
        assert data is not None

        feature_names = dataset.get_dataset().feature_names

        for col in feature_names:

            assert not self.variable_types.get(col)
            if self.feature_configure[col]['ftype'] == 'category':
                self.variable_types[col] = ft.variable_types.Categorical
            elif self.feature_configure[col]['ftype'] == 'numerical':
                self.variable_types[col] = ft.variable_types.Numeric
            elif self.feature_configure[col]['ftype'] == 'bool':
                self.variable_types[col] = ft.variable_types.Boolean
            else:
                assert self.feature_configure[col]['ftype'] == 'datetime'
                self.variable_types[col] = ft.variable_types.Datetime

        es = self.entity_set.entity_from_dataframe(entity_id=self.name, dataframe=data, variable_types=self.variable_types,
                                                   make_index=True, index='data_id')

        primitives = ft.list_primitives()
        trans_primitives = list(primitives[primitives['type'] == 'transform']['name'].values)

        # Create new features using specified primitives
        features, feature_names = ft.dfs(entityset=es, target_entity=self.name,
                                         trans_primitives=trans_primitives)

        dataset.get_dataset().data = features
        dataset.get_dataset().generated_features_names = feature_names

    def _label_encoding(self, dataset: BaseDataset):
        feature_names = dataset.get_dataset().feature_names
        data = dataset.get_dataset().data

        for feature in feature_names:
            if self.feature_configure[feature]['ftype'] == 'category' or self.feature_configure[feature]['ftype'] == 'bool':

                item_label_encoding = LabelEncoder()
                item_label_encoding_model = item_label_encoding.fit(data[feature])
                self.label_encoding[feature] = item_label_encoding_model

                data[feature] = item_label_encoding_model.transform(data[feature])

    def _label_encoding_serialize(self):
        # 序列化label encoding模型字典
        with shelve.open(self.label_encoding_configure_path) as shelve_open:
            shelve_open['label_encoding'] = self.label_encoding
