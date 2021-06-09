import os
import csv

import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle

from utils.bunch import Bunch
from entity.base_dataset import BaseDataset
from utils.Logger import logger


class PlaintextDataset(BaseDataset):
    """
    Reads data
    """

    def __init__(self, name, data_path, task_type, target_name=None, memory_only=True):
        super(PlaintextDataset, self).__init__(name, data_path, task_type, target_name, memory_only)

        assert os.path.isfile(data_path)

        self.type_doc = None
        self.shape = None
        self._bunch = self.load_data()

    def __repr__(self):
        assert self._bunch is not None
        assert self.type_doc is not None
        assert self.get_column_size() > 0 and self.get_row_size() > 0

        self.shape = [self.get_row_size(), self.get_column_size()]

        if self.type_doc in ["csv"]:
            combined_df, _, _ = self._convert_data_dataframe(data=self._bunch.data,
                                                             target=self._bunch.target,
                                                             feature_names=self._bunch.feature_names,
                                                             target_names=self._bunch.target_name)
            if self.shape[0] > self._default_print_size:
                return str(combined_df.head(self._default_print_size))
            else:
                return str(combined_df)

        else:
            combined_df, _, _ = self._convert_data_dataframe(data=self._bunch.data,
                                                             target=self._bunch.target)
            if self.shape[0] > self._default_print_size:
                return str(combined_df.head(self._default_print_size))
            else:
                return str(combined_df)

    def get_dataset(self):
        return self._bunch

    def load_data(self):
        assert "." in self._data_path
        self.type_doc = self._data_path.split(".")[-1]

        assert self.type_doc in ["csv", "libsvm", "txt"]

        data = None
        target = None
        feature_names = None
        target_name = None

        if self.type_doc == "csv":
            try:
                data, target, feature_names, target_name = self.load_csv()
                _, data, target = self._convert_data_dataframe(data=data,
                                                               target=target,
                                                               feature_names=feature_names,
                                                               target_names=target_name)
            except ValueError:
                logger.info(".csv file has object dtype, load_mixed_csv() method has started.")
                data, target, feature_names, target_name = self.load_mixed_csv()
            except IOError:
                logger.info("File path does not exist.")
            finally:
                logger.info(".csv file has been converted to Bunch object.")

            try:
                self.shuffle_data(data, target)
            except TypeError:
                logger.info("CSV file is not read correctly.")

            self._bunch = Bunch(data=data,
                                target=target,
                                target_name=target_name,
                                feature_names=feature_names)

        elif self.type_doc == 'libsvm':
            data, target = self.load_libsvm()
            _, data, target = self._convert_data_dataframe(data=data,
                                                           target=target)
            self.shuffle_data(data, target)
            self._bunch = Bunch(
                data=data,
                target=target
            )

        elif self.type_doc == 'txt':
            data, target = self.load_txt()
            _, data, target = self._convert_data_dataframe(data=data,
                                                           target=target)
            self.shuffle_data(data, target)
            self._bunch = Bunch(
                data=data,
                target=target
            )
        else:
            raise TypeError("File type can not be accepted.")
        return self._bunch

    def load_mixed_csv(self):
        data = pd.read_csv(self._data_path)
        target = data[self._target_name]
        data = data.drop(self._target_name, axis=1)

        feature_names = data.columns
        target_name = self._target_name

        self._row_size = data.shape[0]
        self._column_size = data.shape[1] + target.shape[1]

        return data, target, feature_names, target_name

    def load_csv(self):
        """Loads data from csv_file_name.

        Returns
        -------
        data : Numpy array
            A 2D array with each row representing one sample and each column
            representing the features of a given sample.

        target : Numpy array
            A 1D array holding target variables for all the samples in `data.
            For example target[0] is the target variable for data[0].

        target_names : Numpy array
            A 1D array containing the names of the classifications. For example
            target_names[0] is the name of the target[0] class.
        """
        with open(self._data_path, 'r') as csv_file:

            data_file = csv.reader(csv_file)
            feature_names = next(data_file)
            target_location = -1

            try:
                target_location = feature_names.index(self._target_name)
                target_name = feature_names.pop(target_location)
            except IndexError:
                logger.info("Label is not exist.")
            assert target_name == self._target_name

            self._row_size = n_samples = self.wc_count() - 1
            self._column_size = n_features = len(feature_names)
            data = np.empty((n_samples, n_features))
            target = np.empty((n_samples,), dtype=int)

            for index, row in enumerate(data_file):
                label = row.pop(target_location)
                data[index] = np.asarray(row, dtype=np.float64)
                target[index] = np.asarray(label, dtype=int)

        return data, target, feature_names, self._target_name

    def load_libsvm(self):
        data, target = load_svmlight_file(self._data_path)
        data = data.toarray()
        self._column_size = len(data[0]) + 1
        self._row_size = len(data)
        return data, target

    def load_txt(self):
        target_index = 0
        data = []
        target = []

        with open(self._data_path, 'r') as file:
            lines = file.readlines()

            for index, line_content in enumerate(lines):
                data_index = []
                line_content = line_content.split(' ')

                for column, item in enumerate(line_content):
                    if column != target_index:
                        data_index.append(item)
                    else:
                        target.append(item)

                data_index = list(map(np.float64, data_index))
                data.append(data_index)

            data = np.asarray(data, dtype=np.float64)
            target = list(map(int, target))
            target = np.asarray(target, dtype=int)

            self._column_size = len(data[0]) + 1
            self._row_size = len(data)

            return data, target

    def wc_count(self):
        import subprocess
        out = subprocess.getoutput("wc -l %s" % self._data_path)
        return int(out.split()[0])

    def get_column_size(self):
        return self._column_size

    def get_row_size(self):
        return self._row_size

    def get_target_name(self):
        return self._target_name

    @classmethod
    def _convert_data_dataframe(cls, data, target,
                                feature_names=None, target_names=None):

        data_df = pd.DataFrame(data, columns=feature_names)
        target_df = pd.DataFrame(target, columns=target_names)
        combined_df = pd.concat([data_df, target_df], axis=1)

        return combined_df, data_df, target_df

    @classmethod
    def shuffle_data(cls, data, target):
        return shuffle(data, target)
