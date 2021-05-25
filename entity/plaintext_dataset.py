import csv
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file

from utils.bunch import Bunch


class PlaintextDataset:
    """
    Reads data
    """

    def __init__(self, input_path, target_name="target_name"):
        self.input_path = input_path
        self.target_name = target_name

    def load_data(self):
        type_doc = self.input_path.split(".")[-1]

        if type_doc == "csv":
            data, target, feature_names, target_name = self.load_csv()

            return Bunch(data=data,
                         target=target,
                         target_name=[target_name],
                         feature_names=feature_names)

        elif type_doc == 'libsvm':
            data, target = self.load_libsvm()
            data = data.todense()
            return Bunch(
                data=data,
                target=target
            )

        elif type_doc == 'txt':
            data, target = self.load_txt()
            return Bunch(
                data=data,
                target=target
            )

    def load_csv(self):
        """Loads data from csv_file_name.

        Returns
        -------
        data : Numpy array
            A 2D array with each row representing one sample and each column
            representing the features of a given sample.

        target : Numpy array
            A 1D array holding target variables for all the samples in `data.
            For example target[0] is the target varible for data[0].

        target_names : Numpy array
            A 1D array containing the names of the classifications. For example
            target_names[0] is the name of the target[0] class.
        """
        with open(self.input_path, 'r') as csv_file:

            data_file = csv.reader(csv_file)
            feature_names = next(data_file)
            target_location = -1

            try:
                target_location = feature_names.index(self.target_name)
                target_name = feature_names.pop(target_location)
            except IndexError:
                print("Label is not exist.")
            assert target_name == self.target_name

            n_samples = self.wc_count() - 1
            n_features = len(feature_names)
            data = np.empty((n_samples, n_features))
            target = np.empty((n_samples,), dtype=int)

            for index, row in enumerate(data_file):
                label = row.pop(target_location)
                data[index] = np.asarray(row, dtype=np.float64)
                target[index] = np.asarray(label, dtype=int)

        return data, target, feature_names, self.target_name

    def load_libsvm(self):
        data, target = load_svmlight_file(self.input_path)
        return data, target

    def load_txt(self):
        target_index = 0
        data = []
        target = []

        with open(self.input_path, 'r') as file:
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

            return data, target

    def wc_count(self):
        import subprocess
        out = subprocess.getoutput("wc -l %s" % self.input_path)
        return int(out.split()[0])

    @classmethod
    def _convert_data_dataframe(cls, data, target,
                                feature_names, target_names, sparse_data=False):
        if not sparse_data:
            data_df = pd.DataFrame(data, columns=feature_names)
        else:
            data_df = pd.DataFrame.sparse.from_spmatrix(
                data, columns=feature_names
            )
        target_df = pd.DataFrame(target, columns=[target_names])
        combined_df = pd.concat([data_df, target_df], axis=1)
        X = combined_df[feature_names]
        y = combined_df[target_names]

        return combined_df, X, y
