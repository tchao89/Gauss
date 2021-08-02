# -*- coding: utf-8 -*-
import warnings
import pandas as pd
import tensorflow as tf

from entity.dataset.base_dataset import BaseDataset
from entity.dataset.plain_dataset import PlaintextDataset


class TFPlainDataset(BaseDataset):
    """Gauss_nn Dataset wrapper 
    
    This class is aiming to convert Plain dataset generated by Gauss system
    in memory to tf.train.Dataset which will be used in Gauss_nn system.
    To use the dataset after constructed, user need to build the dataset
    by call the build() function, then call the init() to pass a value to 
    batch_size defined as a tf.placeholder to activate tf.Ops. Finally, all 
    hyper-parameters could be updated by update_dataset_parameters().
    """

    def __init__(self, **params):
        """
        :param file_repeat: bool, if dataset need repeat.
        :param file_repeat_count: int, repeat count times, activate when file_repeat is True.
        :param shuffle_buffer_size: int, count of elements will be filled in buffer.
        :param prefetch_buffer_size: int, count of elements will be prefetched from dataset.
        :param drop_remainder: bool, drop last element when drop_remainder is True and number of elements
                                can not divide batch_size evenly; keep if True.
        """
        if not isinstance(params["dataset"], BaseDataset):
            raise TypeError("dataset must be a instance of PlainDataset.")

        super(TFPlainDataset, self).__init__(name=params["name"],
                                              data_path=None,
                                              task_type=params["task_type"],
                                              target_name=params["target_name"],
                                              memory_only=params["memory_only"])
  
        self._plain_dataset = params["dataset"]

        self._file_repeat = params["file_repeat"] \
            if params.get("file_repeat") else False
        self._file_repeat_count = params["file_repeat_count"] \
            if params.get("file_repeat_count") and params["_file_repeat"] else 1
        self._shuffle_buffer_size = params["shuffle_buffer_size"] \
            if params.get("shuffle_buffer_size") else 100000
        self._prefetch_buffer_size = params["prefetch_buffer_size"] \
            if params.get("prefetch_buffer_size") else 1
        self._drop_remainder = params["drop_remainder"] \
            if params.get("drop_remainder") else True

        # hyper-parameters
        self._batch_size = tf.placeholder(dtype=tf.int64, shape=())
        self._batch_size_param = 8

    def __repr__(self):
        dataset = self._plain_dataset.get_dataset().data.head()
        return str(dataset)


    def build(self):
        self._dataset = self._build_dataset()
        self._iterator = self._dataset.make_initializable_iterator()
        self._next_batch = self._iterator.get_next()

    def update_dataset_parameters(self, **params):
        # TODO: update further hyper parameters will be used in nn.
        """update hyper-parameters generated by auto ml search space"""
        if not params.get("batch_size"):
            raise TypeError("update_dataset_parameters missing 1 required key word parameter: batch_size.")
        self._batch_size_param = params["batch_size"]
        
    def init(self,sess):
        """initialize current iterated tf.Dataset.
        
        activate tf.Operations defined yet by feed batch_size to iterated tf.Dataset,
        batched dataset will be applied to followed steps. 
        """
        self._iterator.initializer.run(session=sess, feed_dict={self._batch_size: self._batch_size_param})


    @property
    def batch_size(self):
        return self._batch_size_param

    @property
    def next_batch(self):
        return self._next_batch

    @property
    def shape(self):
        ori_shape = self._plain_dataset.get_dataset().data.shape 
        return (ori_shape[0], ori_shape[1]+1)
        
    @property
    def info(self):
        return self._plain_dataset.get_dataset().data.info()


    def _build_dataset(self):
        dataset = self._build_raw_dataset(self._plain_dataset)
        dataset = self._repeat_dataset(dataset)
        dataset = self._shuffle_and_batch(dataset)
        dataset = self._apply_prefetch(dataset)
        return dataset

    def _build_raw_dataset(self, dataset: PlaintextDataset) -> tf.data.Dataset:
        """load `PlainDataset` in memory.

        :return : a `tf.data.Dataset` object contains whole PlainDataset contents.
        """
        bunch = dataset.get_dataset()
        data = pd.concat((bunch.data, bunch.target), axis=1)
        dataset = tf.data.Dataset.from_tensor_slices(data.to_dict("list"))
        del bunch
        del data
        return dataset

    def _repeat_dataset(self, dataset):
        """repeat current dataset count times if repeat applied"""
        if self._repeat_dataset:
            dataset = dataset.repeat(self._file_repeat_count)
        return dataset

    def _shuffle_and_batch(self, dataset):
        """randomly sample data from dataset and batch to batch_size."""
        dataset = dataset.shuffle(self._shuffle_buffer_size)
        dataset = dataset.batch(self._batch_size, self._drop_remainder)
        return dataset

    def _apply_prefetch(self, dataset):
        """prefetch for acceleration"""
        dataset = dataset.prefetch(self._prefetch_buffer_size)
        return dataset


    def load_data(self):
        warnings.warn("function load_data will not be used in Gauss_nn module.")

    def get_dataset(self):
        warnings.warn("function get_dataset will not be used in Gauss_nn module.")

    def split(self):
        warnings.warn("function split will not be used in Gauss_nn module.")

    def union(self, dataset: BaseDataset):
        warnings.warn("function union will not be used in Gauss_nn module.")
