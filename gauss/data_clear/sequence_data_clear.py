# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

import copy
import shelve
import yaml
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from gauss.data_clear.base_data_clear import BaseDataClear
from entity.dataset.base_dataset import BaseDataset

from utils.common_component import yaml_read
from utils.Logger import logger
from utils.base import (
    reduce_data, 
    get_current_memory_gb
    )

class SequenceDataClear(BaseDataClear):

        """Fill missing vallues by strategy provided.
        
        Strategy options: 'mean', 'median', 'most_frequent' and 'constant'.
        There are two ways to define the strategy,
        one is by feature type:
        
        Parameters:
        ---------------
        feature_config_path: str; To be loaded config file path.
        
        save_config_path: str; Path for saving processed config. 
        
        save_imputer_path: str; Path for saving serialized imputer.
        
        strategy_dict: dict; Strategy for handle missing values.
        
        missing_values: int, float, str(default); Value for replace missing value. 

        Examples:
        ---------------
        >>> {
                "model": {"name": "ftype"}, 
                "category": {"name": 'most_frequent'}, 
                "numerical": {"name": "mean"}, 
                "bool": {"name": "most_frequent"}, 
                "datetime": {"name": "most_frequent"}
            }
        second is by feature names:
        >>> {
                "model": {"name": "feature"}, 
                "feature 1": {"name": 'most_frequent'}, 
                "feature 2": {"name": 'constant', 
                "fill_value": 0}
            }
        """

        FLOAT = "float"
        FLOAT64 = "float64"
        INT = "int"
        INT64 = "int64"

        CATE = "category"
        NUM = "numerical" 
        DATE = "datetime"
        BOOL = "bool"

        def __init__(self, **params):
            """
            Parameters:
            ----------------
            feature_config_path: str; To be loaded config file path.
            
            save_config_path: str; Path for saving processed config. 
            
            save_imputer_path: str; Path for saving serialized imputer.
            
            strategy_dict: dict; Strategy for handle missing values.
            
            missing_values: int, float, str(default); Value for replace missing value. 
            """

            super(SequenceDataClear, self).__init__(
                name=params["name"],
                train_flag=params["train_flag"],
                enable=params["enable"]
            )
            self._feature_config_path = params["feature_config_path"]
            self._save_config_path = params["save_config_path"] 
            self._save_imputer_path = params["save_imputer_path"]
            self._strategy_dict = params["strategy_dict"] \
                if params.get("strategy_dict") else None
            self._missing_values = params["missing_values"] \
                if params.get("missing_values") else np.nan 
            
            self._default_cate_imputer = SimpleImputer(
                missing_values=self._missing_values,
                strategy="most_frequent"
            )
            self._default_num_imputer = SimpleImputer(
                missing_values=self._missing_values,
                strategy="mean"
            )
            self._feature_imputers = {}
            self._already_data_clear = None


        def _train_run(self, **entity):
            logger.info("Data cleaning status: {flag}".format(flag=self._enable))
            if self._enable:
                self._already_data_clear = True
                self._clean(dataset=entity["dataset"])
            else:
                self._already_data_clear = False
            self.final_configure_generation()
            self._save_serialized()
            if self._feature_imputers is not None:
               del self._feature_imputers

        def _predict_run(self, **entity):
            dataset = entity["dataset"]
            data = dataset.get_dataset().data

            feature_names = dataset.get_dataset().feature_names
            feature_config = yaml_read(self._feature_config_path)
            self._replace_invalid_by_nan(data=data)

            if self._enable:
                with shelve.open(self._save_imputer_path) as file:
                    imputers = file["imputers"]

                for fea_name in feature_names:
                    fea_config = feature_config[fea_name]
                    if imputers.get(fea_name):
                        fea_data = np.array(data[fea_name]).reshape(-1, 1)

                    if self.INT in fea_config["dtype"]:
                        imputers.get(fea_name).fit(fea_data.astype(np.int64))
                    elif self.FLOAT in fea_config["dtype"]:
                        imputers.get(fea_name).fit(fea_data.astype(np.float64))
                    else:
                        imputers.get(fea_name).fit(fea_data)
                    
                    fea_data = imputers.get(fea_name).transform(fea_data)
                    data[fea_name] = fea_data.reshape(1, -1).squeeze(axis=0)

        def _clean(self, dataset: BaseDataset):
            data = dataset.get_dataset().data
            feature_names = dataset.get_dataset().feature_names
            feature_config = yaml_read(self._feature_config_path)

            self._replace_invalid_by_nan(data=data, feature_config=feature_config)

            for fea_name in feature_names:
                fea_data = np.array(data.loc[:,fea_name])
                fea_config = feature_config[fea_name]

                if self._strategy_dict is not None:
                    if self._strategy_dict["model"]["name"] == "ftype":
                        imputer = SimpleImputer(
                            missing_values=self._missing_values,
                            strategy=self._strategy_dict[fea_config["ftype"]]["name"],
                            fill_value=self._strategy_dict[fea_config["ftype"]].\
                                get("fill_value"),
                            add_indicator=True
                        )
                    elif self._strategy_dict["model"]["name"] == "feature":
                        imputer = SimpleImputer(
                            missing_values=self._missing_values,
                            strategy=self._strategy_dict[fea_name]["name"],
                            fill_value=self._strategy_dict[fea_name].\
                                get("fill_value"),
                            add_indicator=True
                        )
                else:
                    if fea_config["ftype"] == self.NUM:
                        imputer = copy.deepcopy(self._default_num_imputer)
                    elif fea_config["ftype"] in [self.CATE, self.BOOL, self.DATE]:
                        imputer = copy.deepcopy(self._default_cate_imputer)
                
                fea_data = fea_data.reshape(-1, 1)

                if self.INT in fea_config["dtype"]:
                    imputer.fit(fea_data.astype(np.int64))
                elif self.FLOAT in fea_config["dtype"]:
                    imputer.fit(fea_data.astype(np.float64))
                else:
                    imputer.fit(fea_data)

                fea_data = imputer.transform(fea_data)
                fea_data = fea_data.reshape(1, -1).squeeze(axis=0)
                self._feature_imputers[fea_name] = imputer
            reduce_data(dataframe=data)

        def _replace_invalid_by_nan(self, data: pd.DataFrame, feature_config):
            for col_name in data.columns:
                dtype = feature_config[col_name]["dtype"]
                is_nan = [self._is_valid(value, dtype)\
                     for value in data.loc[:,col_name]]
                if not all(is_nan):
                    data[:,col_name].where(is_nan, inplace=True)

        def _is_valid(self, value, dtype):
            if dtype == self.INT64:
                try:
                    int(value)
                    return True
                except:
                    return False
            if dtype == self.FLOAT64:
                try:
                    float(value)
                    return True
                except:
                    return False
            return True

        
        def final_configure_generation(self):
            feature_conf = yaml_read(yaml_file=self._feature_config_path)
            with open(self._save_config_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(feature_conf, yaml_file)

        def _save_serialized(self):
            with shelve.open(self._save_imputer_path) as file:
                file["imputers"] = self._feature_imputers


        @property
        def already_data_clear(self):
            return self._already_data_clear