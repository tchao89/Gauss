import yaml

import pandas as pd
from scipy.stats import chi2
from sklearn.feature_selection import chi2, SelectKBest

from gauss.feature_select.base_feature_selector import BaseFeatureSelector
from entity.dataset.base_dataset import BaseDataset
from core import featuretools as ft

from utils.common_component import yaml_write, yaml_read


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

        feature_conf = yaml_read(self._feature_configure_path)
        # remove datetime features.
        for col in data.columns:
            if feature_conf[col]["ftype"] not in ["category", "numerical", "bool"]:
                data.drop([col], axis=1, inplace=True)

        if self._enable is True:
            # target = dataset.get_dataset().target
            if dataset.get_dataset().get("generated_feature_names"):
                feature_names = dataset.get_dataset().generated_feature_names
            else:
                feature_names = dataset.get_dataset().feature_names

            # unsupervised methods will vote for the best features, which will be finished in the future.
            data, generated_features_names = self._ft_method(features=data, feature_names=feature_names)

            dataset.get_dataset().data = data
            dataset.get_dataset().generated_feature_names = list(data.columns)

        self.final_configure_generation(dataset=dataset)

    def _predict_run(self, **entity):
        assert "dataset" in entity.keys()

        dataset = entity['dataset']
        conf = yaml_read(yaml_file=self._final_file_path)

        assert "unsupervised_feature_selector" in conf.keys()
        if conf["unsupervised_feature_selector"] is True:
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

        assert isinstance(self._enable, bool)
        yaml_dict["unsupervised_feature_selector"] = self._enable
        yaml_write(yaml_dict=yaml_dict, yaml_file=self._final_file_path)
