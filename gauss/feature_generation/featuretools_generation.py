"""-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab"""
import gc
import operator

import pandas as pd
import numpy as np

from core import featuretools as ft
from core.featuretools.variable_types.variable import Discrete, Boolean, Numeric

from entity.dataset.base_dataset import BaseDataset
from gauss.feature_generation.base_feature_generation import BaseFeatureGenerator

from utils.Logger import logger
from utils.constant_values import ConstantValues
from utils.yaml_exec import yaml_read
from utils.yaml_exec import yaml_write
from utils.base import get_current_memory_gb


class FeatureToolsGenerator(BaseFeatureGenerator):

    def __init__(self, **params):

        super().__init__(
            name=params["name"],
            train_flag=params["train_flag"],
            enable=params["enable"],
            task_name=params["task_name"],
            feature_configure_path=params["feature_config_path"]
        )

        self._final_file_path = params["final_file_path"]

        self.variable_types = {}
        self.feature_configure = None

        # This is the feature description dictionary, which will generate a yaml file.
        self.yaml_dict = {}

        self.generated_feature_names = None
        self.index_name = "data_id"

    def _train_run(self, **entity):
        assert "train_dataset" in entity.keys()
        dataset = entity["train_dataset"]

        logger.info("Loading Data clearing feature configuration, with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        self._load_feature_configure()
        assert self.feature_configure is not None

        logger.info("Feature generation component flag: " + str(self._enable) + ", with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])

        if self._enable is True:
            self._ft_generator(dataset=dataset)

        logger.info(
            "Feature generation's feature configuration is generating, " + "with current memory usage: %.2f GiB",
            get_current_memory_gb()["memory_usage"])
        self.final_configure_generation(dataset=dataset)

    def _increment_run(self, **entity):
        assert ConstantValues.increment_dataset in entity.keys()
        dataset = entity[ConstantValues.increment_dataset]

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)

        self._load_feature_configure()
        assert self.feature_configure is not None

        if self._enable is True:
            self._ft_generator(dataset=dataset)

    def _predict_run(self, **entity):
        assert "infer_dataset" in entity.keys()
        dataset = entity['infer_dataset']

        data = dataset.get_dataset().data
        assert isinstance(data, pd.DataFrame)

        self._load_feature_configure()
        assert self.feature_configure is not None

        if self._enable is True:
            self._ft_generator(dataset=dataset)

    def _load_feature_configure(self):
        self.feature_configure = yaml_read(self._feature_configure_path)

    def _ft_generator(self, dataset: BaseDataset):
        data = dataset.get_dataset().pop("data")

        assert "data" not in dataset.get_dataset().keys()
        assert data is not None

        feature_names = dataset.get_dataset().feature_names
        for col in feature_names:

            assert not self.variable_types.get(col)
            if self.feature_configure[col]['ftype'] == 'category':
                self.variable_types[col] = ft.variable_types.Categorical
            elif self.feature_configure[col]['ftype'] == 'numerical':
                self.variable_types[col] = ft.variable_types.Numeric
            else:
                assert self.feature_configure[col]['ftype'] == 'datetime'
                self.variable_types[col] = ft.variable_types.Datetime

        logger.info("Featuretools EntitySet object constructs, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        es = ft.EntitySet(id=self.name).entity_from_dataframe(entity_id=self._name, dataframe=data,
                                                              variable_types=self.variable_types,
                                                              make_index=True, index=self.index_name)

        primitives = ft.list_primitives()
        trans_primitives = list(primitives[primitives['type'] == 'transform']['name'].values)

        # pandas method Series.dt.weekofday and Series.dt.week have been deprecated,
        # so featuretools can not use "week" transform method.
        try:
            trans_primitives.remove("week")
        except ValueError:
            logger.info("week transform does not exist in trans_primitives.")
        finally:

            # Create new features using specified primitives
            logger.info("DFS method prepares to start, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            features, feature_names = ft.dfs(entityset=es, target_entity=self.name,
                                             trans_primitives=trans_primitives)

            logger.info("Remove original dataset, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])

            logger.info("Clean dataset method prepares to start, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            generated_data = self.clean_dataset(features)

            logger.info("Clear data and save memory, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])
            del features, data, es
            gc.collect()

            logger.info("Update bunch object and add generated feature names, " + "with current memory usage: %.2f GiB",
                        get_current_memory_gb()["memory_usage"])

            dataset.get_dataset().data = generated_data
            self.generated_feature_names = list(dataset.get_dataset().data.columns)

            retain_features = []
            for index, feature in enumerate(feature_names):
                if feature.name in self.generated_feature_names:
                    retain_features.append(feature)

            dataset.get_dataset().generated_feature_names = retain_features

    def clean_dataset(self, df):
        assert isinstance(df, pd.DataFrame)
        cols = []

        for col in df.columns:
            if df[col].dtype == "object" \
                    or self.index_name in col \
                    or df[col].isin([np.nan, np.inf, -np.inf]).any() is True:
                cols.append(col)

        df.drop(cols, axis=1, inplace=True)
        return df

    def final_configure_generation(self, dataset: BaseDataset):

        if self._enable is True:
            generated_feature_names = dataset.get_dataset().generated_feature_names

            assert operator.eq(list(self.generated_feature_names), list(dataset.get_dataset().data.columns))
            assert operator.eq(list(generated_feature_names), list(self.generated_feature_names))
            assert operator.eq(list(generated_feature_names), list(dataset.get_dataset().data.columns))

            dataset.get_dataset().generated_feature_names = list(self.generated_feature_names)

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
        else:
            for item in self.feature_configure.keys():
                self.feature_configure[item]["used"] = True
            self.yaml_dict = self.feature_configure

        yaml_write(yaml_dict=self.yaml_dict, yaml_file=self._final_file_path)
