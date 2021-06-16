import pandas as pd
from scipy.stats import chi2
from sklearn.feature_selection import chi2, SelectKBest

from gauss.feature_select.base_feature_selector import BaseFeatureSelector
from entity.base_dataset import BaseDataset
from core import featuretools as ft


class UnsupervisedFeatureSelector(BaseFeatureSelector):

    def __init__(self, name, train_flag, enable, feature_config_path, label_encoding_configure_path, feature_select_configure_path):
        """
        :param name:
        :param train_flag:
        :param enable:
        :param feature_config_path:
        :param label_encoding_configure_path:
        :param feature_select_configure_path:
        """
        super(UnsupervisedFeatureSelector, self).__init__(name=name,
                                                          train_flag=train_flag,
                                                          enable=enable,
                                                          feature_configure_path=feature_config_path)

        self.feature_list = []
        self.label_encoding_configure_path = label_encoding_configure_path
        # 特征选择结果配置文件
        self.feature_select_configure_path = feature_select_configure_path

    def _train_run(self, **entity):
        """

        :param entity:
        :return:
        """
        assert "dataset" in entity.keys()
        dataset = entity['dataset']

        assert isinstance(dataset, BaseDataset)

        data = dataset.get_dataset().data
        target = dataset.get_dataset().target
        generated_features_names = dataset.get_dataset().generated_feature_names

        # unsupervised methods will vote for the best features, which will be finished in the future.
        data, generated_features_names = self._ft_method(features=data, feature_names=generated_features_names)

        dataset.get_dataset().data = data
        dataset.get_dataset().generated_feature_names = generated_features_names

    def _predict_run(self, **entity):
        assert "dataset" in entity.keys()
        pass

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

