"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
"""
import os.path
import shutil

from gauss_factory.gauss_factory_producer import GaussFactoryProducer
from utils.base import mkdir
from utils.yaml_exec import yaml_read, yaml_write


class UpdateBest:
    def __init__(self, main_work_root):
        self.__main_work_root = main_work_root
        self.__dispatch_file_name = "dispatch_configure.yaml"

        self.__dispatch_configure = None
        self.__model_zoo = None

    def run(self):
        """
        This method will get best child-task first, then
         copy temp generated folder of all kinds models to main work root.
        And finally, this method will reconstruct main work root.
        :return: None
        """
        pass

    def __load_dispatch_configure(self):
        """
        Get generated ids.
        :return:
        """
        self.__dispatch_configure = yaml_read(
            os.path.join(self.__main_work_root,
                         self.__dispatch_file_name))

    def __load_train_user_configure(self):
        """
        Get model zoo value.
        :return:
        """
        pass

    def __load_pipeline_configure(self):
        """
        Get metric result name and metric value
        :return:
        """
        pass

    def __get_best_root(self):
        pass

    def __delete_generated_folder(self):
        pass

    def __copy_folder(self, source_path, target_path):
        """
        Copy all files in source_path to target_path
        :param source_path: source file path
        :param target_path: target file path
        :return: None
        """
        if not os.path.exists(target_path):
            mkdir(target_path)
        # 输出path_read目录下的所有文件包括文件夹的名称
        names = os.listdir(source_path)
        # 循环遍历所有的文件或文件夹
        for name in names:
            # 定义新的读入路径（就是在原来目录下拼接上文件名）
            path_read_new = os.path.join(source_path, name)
            # 定义新的写入路径（就是在原来目录下拼接上文件名）
            path_write_new = os.path.join(target_path, name)
            # 判断该读入路径是否是文件夹，如果是文件夹则执行递归，如果是文件则执行复制操作
            if os.path.isdir(path_read_new):
                # 判断写入路径中是否存在该文件夹，如果不存在就创建该文件夹
                if not os.path.exists(path_write_new):
                    # 创建要写入的文件夹
                    os.mkdir(path_write_new)
                # 执行递归函数，将文件夹中的文件复制到新创建的文件夹中（保留原始目录结构）
                self.__copy_folder(path_read_new, path_write_new)
            else:
                # 将文件path_read_new复制到path_write_new
                shutil.copyfile(path_read_new, path_write_new)

    def __reconstruct_folder(self, folder, init_prefix=None):
        """
        This method is used to reconstruct a root folder,
        which can solve path problem where copy_folder() method brings.
        :param folder: new folder that copy_folder() method brings.
        :param init_prefix: initial prefix
        :return: None
        """
        if init_prefix is None:
            raise ValueError("Value: init_prefix can not be None.")

        if not isinstance(folder, str):
            raise ValueError("Value: folder: {} is not a correct root path.".format(
                folder))

        assert os.path.isdir(folder), "Value: folder:{} is not exist.".format(
            folder)

        folder_path = os.walk(folder)
        prefix = os.path.join(folder).split("/")[-1]

        for path, dir_list, file_list in folder_path:
            for file_name in file_list:
                file_path = os.path.join(path, file_name)

                if file_path.split(".")[-1] == "yaml":
                    configure_dict = yaml_read(os.path.join(path, file_name))
                    self.__replace_path(source_dict=configure_dict,
                                        init_prefix=init_prefix,
                                        prefix=prefix)

                    yaml_write(yaml_dict=configure_dict, yaml_file=file_path)

    @classmethod
    def __replace_path(cls, source_dict, init_prefix=None, prefix=None):
        """
        This method will replace folder name in a json dict by recursion method.
        :param source_dict: origin dict object.
        :param init_prefix: a string object in initial dict.
        :param prefix: new string object used to replace initial string object.
        :return: None
        """
        if not isinstance(init_prefix, str) and not isinstance(prefix, str):
            raise ValueError("Value: init_prefix: {} and prefix: {} must be string.".format(
                init_prefix, prefix
            ))
        if init_prefix is None or prefix is None:
            raise ValueError("Value: init_prefix:{} and prefix:{} can not be None.".format(
                init_prefix, prefix
            ))
        if isinstance(source_dict, dict):
            for key in source_dict.keys():
                if isinstance(source_dict[key], str):
                    source_dict[key] = source_dict[key].replace(init_prefix, prefix)
                if isinstance(source_dict[key], dict):
                    cls.__replace_path(
                        source_dict=source_dict[key],
                        init_prefix=init_prefix,
                        prefix=prefix
                    )

    @classmethod
    def _create_entity(cls, entity_name: str, **params):
        gauss_factory = GaussFactoryProducer()
        entity_factory = gauss_factory.get_factory(choice="entity")
        return entity_factory.get_entity(entity_name=entity_name, **params)
