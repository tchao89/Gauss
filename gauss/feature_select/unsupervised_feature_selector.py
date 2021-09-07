
import pandas as pd
from scipy.stats import chi2
from sklearn.feature_selection import chi2, SelectKBest

from gauss.feature_select.base_feature_selector import BaseFeatureSelector
from entity.dataset.base_dataset import BaseDataset
from core.featuretools import variable_types

from utils.common_component import yaml_write, yaml_read, feature_list_generator
from utils.Logger import logger
from utils.base import get_current_memory_gb


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
                                                          task_name=params["task_name"],
                                                          feature_configure_path=params["feature_config_path"])

        self.feature_list = []
        self._label_encoding_configure_path = params["label_encoding_configure_path"]
        self._final_file_path = params["final_file_path"]

        self._feature_conf = None

    def _train_run(self, **entity):
        """

        :param entity:
        :return:
        """
        assert "train_dataset" in entity.keys()
        dataset = entity['train_dataset']
        assert isinstance(dataset, BaseDataset)
        data = dataset.get_dataset().data

        logger.info("Unsupervised feature selector component flag: " + str(self._enable))
        self._feature_conf = yaml_read(self._feature_configure_path)

        logger.info("Remove datetime features, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])
        for col in data.columns:
            if self._feature_conf[col]["ftype"] not in ["category", "numerical", "bool"]:
                self._feature_conf[col]["used"] = False
                data.drop([col], axis=1, inplace=True)

        logger.info("Starting unsupervised feature selecting, method: featuretools feature selection, " + "with current memory usage: %.2f GiB",
                    get_current_memory_gb()["memory_usage"])

        if self._enable is True:
            self._ft_method(features=data)

        dataset.get_dataset().generated_feature_names = list(data.columns)
        self.final_configure_generation(dataset=dataset)

    def _predict_run(self, **entity):
        assert "infer_dataset" in entity.keys()

        dataset = entity['infer_dataset']
        conf = yaml_read(yaml_file=self._final_file_path)

        if self._enable is True:
            generated_feature_names = feature_list_generator(feature_conf=conf)
            dataset.get_dataset().data = dataset.get_dataset().data[generated_feature_names]
            dataset.get_dataset().generated_feature_names = generated_feature_names

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

    def _ft_method(self, features: pd.DataFrame):
        """
        :param features:
        :return:
        """

        logger.info("Starting to remove low information features, " + "with current memory usage: %.2f GiB, total features: %d",
                    get_current_memory_gb()["memory_usage"], len(features.columns))
        self.remove_low_information_features(features)

        logger.info("Starting to remove highly null features, " + "with current memory usage: %.2f GiB, total features: %d",
                    get_current_memory_gb()["memory_usage"], len(features.columns))
        self.remove_highly_null_features(features)

        logger.info("Starting to remove single value features, " + "with current memory usage: %.2f GiB, total features: %d",
                    get_current_memory_gb()["memory_usage"], len(features.columns))
        self.remove_single_value_features(features)

        # remove remove_highly_correlated_features() method.
        # logger.info("Starting to remove_highly_correlated_features, " + "with current memory usage: %.2f GiB",
        #                     get_current_memory_gb()["memory_usage"])
        # self.remove_highly_correlated_features(features)
        logger.info("Unsupervised feature selecting has finished, " + "with current memory usage: %.2f GiB, total features: %d",
                    get_current_memory_gb()["memory_usage"], len(features.columns))

    def final_configure_generation(self, dataset: BaseDataset):
        if self._enable:
            generated_feature_names = dataset.get_dataset().get("generated_feature_names")
            assert generated_feature_names is not None

            for item in list(self._feature_conf.keys()):
                if item not in generated_feature_names:
                    self._feature_conf.pop(item)

        yaml_write(yaml_dict=self._feature_conf, yaml_file=self._final_file_path)

    @classmethod
    def remove_low_information_features(cls, feature_matrix):
        """Select features that have at least 2 unique values and that are not all null

            Args:
                feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances

            Returns:
                (feature_matrix, features)

        """
        features = []

        columns = feature_matrix.columns
        for feature in columns:
            if feature_matrix[feature].nunique(dropna=False) > 1 and feature_matrix[feature].dropna().shape[0] > 0:
                features.append(feature)
        features = list(set(columns).difference(set(features)))
        feature_matrix.drop(features, axis=1, inplace=True)

    @classmethod
    def remove_highly_null_features(cls, feature_matrix, pct_null_threshold=0.95):
        """
            Removes columns from a feature matrix that have higher than a set threshold
            of null values.

            Args:
                feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances.
                pct_null_threshold (float): If the percentage of NaN values in an input feature exceeds this amount,
                        that feature will be considered highly-null. Defaults to 0.95.

            Returns:
                pd.DataFrame, list[:class:`.FeatureBase`]:
                    The feature matrix and the list of generated feature definitions. Matches dfs output.
                    If no feature list is provided as input, the feature list will not be returned.
        """
        if pct_null_threshold < 0 or pct_null_threshold > 1:
            raise ValueError("pct_null_threshold must be a float between 0 and 1, inclusive.")
        columns = feature_matrix.columns
        percent_null_by_col = {}
        for feature in columns:
            percent_null_by_col[feature] = feature_matrix[feature].isnull().mean()

        if pct_null_threshold == 0.0:
            keep = [f_name for f_name, pct_null in percent_null_by_col.items()
                    if pct_null <= pct_null_threshold]
        else:
            keep = [f_name for f_name, pct_null in percent_null_by_col.items()
                    if pct_null < pct_null_threshold]

        features = list(set(columns).difference(set(keep)))
        feature_matrix.drop(features, axis=1, inplace=True)

    @classmethod
    def remove_single_value_features(cls, feature_matrix, count_nan_as_value=False):
        """Removes columns in feature matrix where all the values are the same.

            Args:
                feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances.
                count_nan_as_value (bool): If True, missing values will be counted as their own unique value.
                            If set to False, a feature that has one unique value and all other
                            data missing will be removed from the feature matrix. Defaults to False.

             Returns:
                pd.DataFrame, list[:class:`.FeatureBase`]:
                    The feature matrix and the list of generated feature definitions.
                    Matches dfs output.
                    If no feature list is provided as input, the feature list will not be returned.
        """
        columns = feature_matrix.columns
        unique_counts_by_col = {}
        for feature in columns:
            unique_counts_by_col[feature] = feature_matrix[feature].nunique(dropna=not count_nan_as_value)

        keep = [f_name for f_name, unique_count
                in unique_counts_by_col.items() if unique_count > 1]
        features = list(set(columns).difference(set(keep)))
        feature_matrix.drop(features, axis=1, inplace=True)

    @classmethod
    def remove_highly_correlated_features(cls, feature_matrix, pct_corr_threshold=0.95,
                                          features_to_check=None, features_to_keep=None):

        """Removes columns in feature matrix that are highly correlated with another column.

            Note:
                We make the assumption that, for a pair of features, the feature that is further
                right in the feature matrix produced by ``dfs`` is the more complex one.
                The assumption does not hold if the order of columns in the feature
                matrix has changed from what ``dfs`` produces.

            Args:
                feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature
                            names and rows are instances.
                pct_corr_threshold (float): The correlation threshold to be considered highly
                            correlated. Defaults to 0.95.
                features_to_check (list[str], optional): List of column names to check
                            whether any pairs are highly correlated. Will not check any
                            other columns, meaning the only columns that can be removed
                            are in this list. If null, defaults to checking all columns.
                features_to_keep (list[str], optional): List of column names to keep even
                            if correlated to another column. If null, all columns will be
                            candidates for removal.

            Returns:
                pd.DataFrame, list[:class:`.FeatureBase`]:
                    The feature matrix and the list of generated feature definitions.
                    Matches dfs output. If no feature list is provided as input,
                    the feature list will not be returned. For consistent results,
                    do not change the order of features outputted by dfs.
        """
        if pct_corr_threshold < 0 or pct_corr_threshold > 1:
            raise ValueError("pct_corr_threshold must be a float between 0 and 1, inclusive.")

        if features_to_check is None:
            features_to_check = feature_matrix.columns
        else:
            for f_name in features_to_check:
                assert f_name in feature_matrix.columns, "feature named {} is not in feature matrix".format(f_name)

        if features_to_keep is None:
            features_to_keep = []

        pandas_numerics = variable_types.PandasTypes.unsupervised_pandas_numerics
        numeric_and_boolean_dtypes = pandas_numerics

        fm_to_check = (feature_matrix[features_to_check]).select_dtypes(
            include=numeric_and_boolean_dtypes)

        dropped = set()
        columns_to_check = fm_to_check.columns
        # When two features are found to be highly correlated,
        # we drop the more complex feature
        # Columns produced later in dfs are more complex
        for i in range(len(columns_to_check) - 1, 0, -1):
            more_complex_name = columns_to_check[i]
            more_complex_col = fm_to_check[more_complex_name]

            for j in range(i - 1, -1, -1):
                less_complex_name = columns_to_check[j]
                less_complex_col = fm_to_check[less_complex_name]

                if abs(more_complex_col.corr(less_complex_col)) >= pct_corr_threshold:
                    dropped.add(more_complex_name)
                    break

        keep = [f_name for f_name in feature_matrix.columns
                if (f_name in features_to_keep or f_name not in dropped)]
        features = list(set(feature_matrix.columns).difference(set(keep)))
        feature_matrix.drop(features, axis=1, inplace=True)
