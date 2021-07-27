# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
import shelve
import operator

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from core import featuretools as ft
from core.featuretools.variable_types.variable import Discrete, Boolean, Numeric

from entity.dataset.base_dataset import BaseDataset
from gauss.feature_generation.base_feature_generation import BaseFeatureGenerator
from utils.Logger import logger
from utils.common_component import yaml_read, yaml_write


class FeatureToolsGenerator(BaseFeatureGenerator):

    def __init__(self, **params):

        super(FeatureToolsGenerator, self).__init__(name=params["name"],
                                                    train_flag=params["train_flag"],
                                                    enable=params["enable"],
                                                    feature_configure_path=params["feature_config_path"])

        self.entity_set = ft.EntitySet(id=self.name)
        self._label_encoding_configure_path = params["label_encoding_configure_path"]
        self.label_encoding = {}
        self._final_file_path = params["final_file_path"]

        self.variable_types = {}
        self.feature_configure = None

        # This is the feature description dictionary, which will generate a yaml file.
        self.yaml_dict = {}

        self.generated_feature_names = None
        self.index_name = "data_id"

    def _train_run(self, **entity):
        assert "dataset" in entity.keys()
        dataset = entity["dataset"]
        self._set_feature_configure()
        assert self.feature_configure is not None
        self._label_encoding(dataset=dataset)
        self._label_encoding_serialize()

        if self._enable:
            self._ft_generator(dataset=dataset)

        self.final_configure_generation(dataset=dataset)

    def _predict_run(self, **entity):

        assert "dataset" in entity.keys()
        dataset = entity['dataset']

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)

        feature_names = dataset.get_dataset().feature_names

        self._set_feature_configure()
        assert self.feature_configure is not None

        with shelve.open(self._label_encoding_configure_path) as shelve_open:
            le_model_list = shelve_open['label_encoding']

            for col in feature_names:
                if self.feature_configure[col]['ftype'] == "category" or self.feature_configure[col]['ftype'] == "bool":
                    assert le_model_list.get(col)
                    le_model = le_model_list[col]

                    label_dict = dict(zip(le_model.classes_, le_model.transform(le_model.classes_)))
                    status_list = data[col].unique().tolist()

                    for item in status_list:
                        if label_dict.get(item) is None:
                            logger.info(
                                "feature: " + str(col) + " has an abnormal value (unseen by label encoding): " + str(
                                    item))
                            raise ValueError("feature: " + str(
                                col) + " has an abnormal value (unseen by label encoding): " + str(item))

                    data[col] = le_model.transform(data[col])
        if self._enable is True:
            self._ft_generator(dataset=dataset)

    def _set_feature_configure(self):
        self.feature_configure = yaml_read(self._feature_configure_path)

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

        es = self.entity_set.entity_from_dataframe(entity_id=self.name, dataframe=data,
                                                   variable_types=self.variable_types,
                                                   make_index=True, index=self.index_name)

        primitives = ft.list_primitives()
        trans_primitives = list(primitives[primitives['type'] == 'transform']['name'].values)

        # pandas method Series.dt.weekofday and Series.dt.week have been deprecated,
        # so featuretools can not use "week" transform method.
        try:
            trans_primitives.remove("week")
        except ValueError:
            logger.info("week transform does not exist in trans_primitives")
        finally:

            # Create new features using specified primitives
            features, feature_names = ft.dfs(entityset=es, target_entity=self.name,
                                             trans_primitives=trans_primitives)

            data = self.clean_dataset(features)

            dataset.get_dataset().data = data
            # raw generated features.
            dataset.get_dataset().generated_feature_names = feature_names

            self.generated_feature_names = list(data.columns)

    def _label_encoding(self, dataset: BaseDataset):
        feature_names = dataset.get_dataset().feature_names
        data = dataset.get_dataset().data

        for feature in feature_names:
            if self.feature_configure[feature]['ftype'] == 'category' or \
                    self.feature_configure[feature]['ftype'] == 'bool':
                item_label_encoding = LabelEncoder()
                item_label_encoding_model = item_label_encoding.fit(data[feature])
                self.label_encoding[feature] = item_label_encoding_model

                data[feature] = item_label_encoding_model.transform(data[feature])

    def _label_encoding_serialize(self):
        # 序列化label encoding模型字典
        with shelve.open(self._label_encoding_configure_path) as shelve_open:
            shelve_open['label_encoding'] = self.label_encoding

    def clean_dataset(self, df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

        for col in df.columns:
            if df[col].dtype == "object" or self.index_name in col:
                df.drop([col], axis=1, inplace=True)

        df.dropna(axis=1, inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any()
        features = indices_to_keep[indices_to_keep == True].index
        return df[features].astype(np.float64)

    def final_configure_generation(self, dataset: BaseDataset):

        if self._enable:
            generated_feature_names = dataset.get_dataset().generated_feature_names
            assert operator.eq(list(self.generated_feature_names), list(dataset.get_dataset().data.columns))

            retain_features = []
            for index, feature in enumerate(generated_feature_names):
                if feature.name in self.generated_feature_names:
                    retain_features.append(feature)

            dataset.get_dataset().generated_feature_names = retain_features
            generated_feature_names = dataset.get_dataset().generated_feature_names

            assert operator.eq(list(generated_feature_names), list(self.generated_feature_names))
            assert operator.eq(list(generated_feature_names), list(dataset.get_dataset().data.columns))

            for index, feature in enumerate(generated_feature_names):

                if issubclass(feature.variable_type, Discrete):
                    ftype = "category"
                    dtype = "int64"
                elif issubclass(feature.variable_type, Boolean):
                    ftype = "bool"
                    dtype = "int64"
                elif issubclass(feature.variable_type, Numeric):
                    ftype = "numerical"
                    dtype = "float64"
                else:
                    raise ValueError("Unknown input feature ftype: " + str(feature.name))

                item_dict = {"name": feature.name, "index": index, "dtype": dtype, "ftype": ftype, "used": True}
                assert feature.name not in self.yaml_dict.keys()
                if feature.name in self.generated_feature_names:
                    self.yaml_dict[feature.name] = item_dict
                dataset.get_dataset().generated_feature_names = list(self.generated_feature_names)
        else:

            for item in self.feature_configure.keys():
                self.feature_configure[item]["used"] = True
            self.yaml_dict = self.feature_configure

        yaml_write(yaml_dict=self.yaml_dict, yaml_file=self._final_file_path)
