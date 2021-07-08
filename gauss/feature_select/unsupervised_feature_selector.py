import yaml

import pandas as pd
from scipy.stats import chi2
from sklearn.feature_selection import chi2, SelectKBest

from gauss.feature_select.base_feature_selector import BaseFeatureSelector
from entity.dataset.base_dataset import BaseDataset
from core import featuretools as ft


class UnsupervisedFeatureSelector(BaseFeatureSelector):

    def __init__(self, **params):

        """
        :param name:
        :param train_flag:
        :param enable:
        :param feature_config_path:
        :param label_encoding_configure_path:
        :param feature_select_configure_path:
        """
        super(UnsupervisedFeatureSelector, self).__init__(name=params["name"],
                                                          train_flag=params["train_flag"],
                                                          enable=params["enable"],
                                                          feature_configure_path=params["feature_config_path"])

        self.feature_list = []
        self._label_encoding_configure_path = params["label_encoding_configure_path"]
        self._final_file_path = params["final_file_path"]

    def _train_run(self, **entity):
        """

        :param entity:
        :return:
        """
        assert "dataset" in entity.keys()
        dataset = entity['dataset']

        assert isinstance(dataset, BaseDataset)

        data = dataset.get_dataset().data
        # target = dataset.get_dataset().target
        generated_features_names = dataset.get_dataset().generated_feature_names

        # unsupervised methods will vote for the best features, which will be finished in the future.
        data, generated_features_names = self._ft_method(features=data, feature_names=generated_features_names)

        dataset.get_dataset().data = data
        dataset.get_dataset().generated_feature_names = list(data.columns)
        self.final_configure_generation(dataset=dataset)

    def _predict_run(self, **entity):
        assert "dataset" in entity.keys()

        dataset = entity['dataset']
        print(self._final_file_path)

        conf_file = open(self._final_file_path, 'r', encoding='utf-8')
        conf = conf_file.read()
        conf = yaml.load(conf, Loader=yaml.FullLoader)
        conf_file.close()

        generated_feature_names = list(conf.keys())
        dataset.get_dataset().data = dataset.get_dataset().data[generated_feature_names]
        dataset.get_dataset().generated_feature_names = dataset.get_dataset().generated_feature_names

    @classmethod
    def _chi2_method(cls, features, target, k):
        """ Compute chi-squared stats between each non-negative feature and class.
        This method should just be used for classification.

        :param features: features for dataset.
        :param target: labels for dataset.
        :return:array-like data structure depends on input. notes: return values will override BaseDataset object
        """
        features = SelectKBest(chi2, k=k).fit_transform(features, target)

        return features

    @classmethod
    def _ft_method(cls, features: pd.DataFrame, feature_names: list):
        """
        :param features:
        :param feature_names:
        :return:
        """
        features, feature_names = ft.selection.remove_low_information_features(features, feature_names)
        features, feature_names = ft.selection.remove_highly_correlated_features(features, feature_names)
        features, feature_names = ft.selection.remove_highly_null_features(features, feature_names)
        features, feature_names = ft.selection.remove_single_value_features(features, feature_names)
        return features, feature_names

    def final_configure_generation(self, dataset: BaseDataset):

        feature_conf_file = open(self._feature_configure_path, 'r', encoding='utf-8')
        feature_conf = feature_conf_file.read()
        feature_conf = yaml.load(feature_conf, Loader=yaml.FullLoader)
        feature_conf_file.close()

        yaml_dict = {}
        data = dataset.get_dataset().data

        for col_name in data.columns:
            yaml_dict[col_name] = feature_conf[col_name]

        with open(self._final_file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(yaml_dict, yaml_file)
