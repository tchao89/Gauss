"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
"""
import os.path
import shutil

from gauss_factory.gauss_factory_producer import GaussFactoryProducer
from utils.base import mkdir
from utils.bunch import Bunch
from utils.constant_values import ConstantValues
from utils.yaml_exec import yaml_read, yaml_write


class UpdateBestApplication:
    def __init__(self, main_work_root):
        self.__main_work_root = main_work_root
        self.__dispatch_file_name = "dispatch_configure.yaml"
        self.__train_user_configure_file_name = "train_user_config.yaml"
        self.__pipeline_configure_file_name = "pipeline_configure.yaml"
        self.__success_file_name = "success.yaml"
        self.__dispatch_configure = None
        self.__train_user_configure = None
        self.__model_zoo = None
        self.__metric_name = None
        self.__metric_value = None
        self.__optimize_mode = None
        self.__work_space = None
        self.__generated_ids = None
        self.__main_id = None
        self.__model_num = 0
        self.__docker_num = 0

        self.__evaluate_result = Bunch()

    def run(self):
        """
        This method will get best child-task first, then
         copy temp generated folder of all kinds models to main work root.
        And finally, this method will reconstruct main work root.
        :return: None
        """
        self.__load_dispatch_configure()
        self.__load_train_user_configure()

        if self.__metric_name is not None:
            metric_params = Bunch(name=self.__metric_name)
            metric = self.__create_entity(entity_name=self.__metric_name,
                                          **metric_params)
        else:
            raise TypeError(
                "Value: metric name should be type of str, but get {} instead.".format(
                    self.__metric_name))

        for generated_id in self.__generated_ids:
            success_flag = self.__load_success_flag(generated_id)
            if success_flag:
                pipeline_configure = self.__load_pipeline_configure(generated_id)
                model_name = list(pipeline_configure[ConstantValues.model].keys())[0]
                model_dict = pipeline_configure[ConstantValues.model][model_name]
                if model_name in self.__model_zoo:
                    metric_result = metric.reconstruct(metric_value=model_dict[ConstantValues.metric_result])
                    if self.__evaluate_result.get(model_name) is None:
                        self.__evaluate_result[model_name] = {"metric_result": metric_result, "id": generated_id}
                    else:
                        if self.__evaluate_result.get(model_name).get(ConstantValues.metric_result).__cmp__(
                                metric_result) < 0:
                            self.__evaluate_result[model_name] = {"metric_result": metric_result, "id": generated_id}
                else:
                    raise ValueError("Value: {} is not in model zoo.".format(model_name))
            else:
                raise RuntimeError("An unexpected error happened in generated id: {}.".format(generated_id))

        for model_name in self.__evaluate_result.keys():
            eval_dict = self.__evaluate_result[model_name]

            generated_id = eval_dict["id"]
            pipeline_configure = self.__load_pipeline_configure(generated_id)

            folder_path = os.path.join(self.__work_space, generated_id)
            folder_path = os.path.join(folder_path, model_name)

            yaml_write(yaml_dict=pipeline_configure,
                       yaml_file=os.path.join(folder_path, self.__pipeline_configure_file_name))
            self.__copy_folder(source_path=folder_path, target_path=os.path.join(self.__main_work_root, model_name))
            self.__reconstruct_folder(folder=os.path.join(self.__main_work_root, model_name),
                                      init_prefix=generated_id)

        # for generated_id in self.__generated_ids:
        #     self.__delete_generated_folder(generated_id)

    def __load_success_flag(self, generated_id):
        temp_root = os.path.join(self.__work_space, generated_id)
        file_path = os.path.join(temp_root, self.__success_file_name)
        return os.path.exists(file_path)

    def __load_dispatch_configure(self):
        """
        Load dispatch configure and get generated ids.
        :return: None
        """
        self.__dispatch_configure = yaml_read(
            os.path.join(self.__main_work_root,
                         self.__dispatch_file_name))
        self.__docker_num = self.__dispatch_configure["docker_num"]
        self.__generated_ids = self.__dispatch_configure["generated_ids"]
        self.__work_space = self.__dispatch_configure["work_space"]
        self.__main_id = self.__dispatch_configure["main_id"]

    def __load_train_user_configure(self):
        """
        Get model zoo value.
        :return:
        """
        file_path = os.path.join(self.__main_work_root,
                                 self.__train_user_configure_file_name)
        self.__train_user_configure = yaml_read(file_path)
        self.__model_zoo = self.__train_user_configure["model_zoo"]
        self.__metric_name = self.__train_user_configure["metric_name"]

    def __load_pipeline_configure(self, generated_id):
        """
        Load pipeline dict from temp id.
        :return: pipeline configure dict.
        """
        temp_root = os.path.join(self.__work_space, generated_id)
        file_path = os.path.join(temp_root, self.__pipeline_configure_file_name)
        return yaml_read(file_path)

    def __delete_generated_folder(self, temp_id):
        folder_path = os.path.join(self.__work_space, temp_id)
        shutil.rmtree(folder_path)

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

        if not os.path.isdir(folder):
            mkdir(folder)

        folder_path = os.walk(folder)
        prefix = os.path.join(folder).split("/")[-1]

        for path, dir_list, file_list in folder_path:
            for file_name in file_list:
                file_path = os.path.join(path, file_name)

                if file_path.split(".")[-1] == "yaml":
                    configure_dict = yaml_read(os.path.join(path, file_name))
                    self.__replace_path(source_dict=configure_dict,
                                        init_prefix=init_prefix,
                                        prefix=self.__main_id)

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
    def __create_component(cls, component_name: str, **params):
        gauss_factory = GaussFactoryProducer()
        component_factory = gauss_factory.get_factory(choice="component")
        return component_factory.get_component(component_name=component_name, **params)

    @classmethod
    def __create_entity(cls, entity_name: str, **params):
        gauss_factory = GaussFactoryProducer()
        entity_factory = gauss_factory.get_factory(choice="entity")
        return entity_factory.get_entity(entity_name=entity_name, **params)


if __name__ == "__main__":
    application = UpdateBestApplication(main_work_root="/home/liangqian/Gauss/experiments/j97qq5")
    application.run()
