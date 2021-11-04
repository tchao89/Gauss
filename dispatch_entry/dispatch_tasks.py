"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
"""
import argparse
import os.path

from jinja2 import Template

from utils.yaml_exec import yaml_read
from utils.yaml_exec import yaml_write
from utils.constant_values import ConstantValues
from utils.bunch import Bunch


class DispatchTasks:
    def __init__(self,
                 user_configure_path: str = None,
                 system_configure_path: str = None):
        self.__tmpl_fun = """
# -*- coding: utf-8 -*-

# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
from pipeline.dispatch_pipeline.dispatch_udf_modeling_graph import UdfModelingGraph

def {{fun}}(name="udf", user_configure=None, system_configure=None):
    if user_configure is None:
        user_configure = {{user_configure}}

    if system_configure is None:
        system_configure = {{system_configure}}

    model_graph = UdfModelingGraph(name=name,
                                   user_configure=user_configure,
                                   system_configure=system_configure)

    model_graph.run()


{{fun}}()
        """

        self.__user_configure_path = user_configure_path
        if not os.path.isfile(self.__user_configure_path):
            raise NotADirectoryError(
                "Path: {} is not a correct user configure path.".format(
                    self.__user_configure_path))

        self.__system_configure_path = system_configure_path
        if not os.path.isfile(self.__system_configure_path):
            raise NotADirectoryError(
                "Path: {} is not a correct system configure path.".format(
                    self.__system_configure_path))

        self.__dispatch_configure = Bunch()
        self.__work_space = None
        self.__main_id = None

    def run(self):
        user_configure = yaml_read(self.__user_configure_path)
        user_configure = Bunch(**user_configure)

        system_configure = yaml_read(self.__system_configure_path)
        system_configure = Bunch(**system_configure)

        try:
            mode = user_configure.mode
        except KeyError:
            raise KeyError("Key: mode is not in user configure.")

        if mode == "auto":
            self.__auto_dispatch(user_configure, system_configure)
        elif mode == "udf":
            self.__udf_dispatch(user_configure, system_configure)
        else:
            raise ValueError("Value: pipeline_configure.mode is illegal.")

        self.__generated_dispatch_configure()

    def __auto_dispatch(self, user_configure, system_configure):
        raise ValueError("If train mode is `auto`, dispatch method can not be used.")

    def __udf_dispatch(self, user_configure, system_configure):
        self.__main_id = os.path.split(user_configure[ConstantValues.work_root])[1]
        self.__work_space = os.path.split(user_configure[ConstantValues.work_root])[0]
        docker_num = len(user_configure.model_zoo) * len(system_configure.keys())

        if docker_num <= 0:
            raise ValueError("Value: docker_num is {}, but it should be more than zero.".format(docker_num))
        docker_num = len(user_configure.model_zoo) * len(system_configure.keys())

        if docker_num <= 0:
            raise ValueError("Value: docker_num is {}, but it should be more than zero.".format(docker_num))
        self.__dispatch_configure.docker_num = docker_num

        generated_ids = []
        for model_name in user_configure.model_zoo:
            for index, system_key in enumerate(system_configure):
                generated_id = self.__main_id + "-" + model_name + "-" + str(index)
                generated_ids.append(generated_id)

                temp_work_root = os.path.join(self.__work_space, generated_id)
                user_configure[ConstantValues.work_root] = temp_work_root

                train_func = self.__render(self.__tmpl_fun,
                                           fun="main_train",
                                           user_configure=user_configure,
                                           system_configure=system_configure[system_key])

                self.__save(code_text=train_func, filename="./dispatch_methods/" + generated_id + ".py")
        self.__dispatch_configure.docker_num = docker_num
        self.__dispatch_configure.generated_ids = generated_ids
        self.__dispatch_configure.main_id = self.__main_id
        self.__dispatch_configure.work_space = self.__work_space

    def __generated_dispatch_configure(self):
        yaml_dict = dict(self.__dispatch_configure)
        if self.__main_id is not None and self.__work_space is not None:
            work_root = os.path.join(self.__work_space, self.__main_id)
            yaml_write(yaml_dict=yaml_dict,
                       yaml_file=os.path.join(work_root, "dispatch_configure.yaml"))

    @classmethod
    def __render(cls, tmpl, **params):
        """
        :param tmpl: template code.
        :param params: params used in template code.
        :return: code text.
        """
        params = dict(**params)
        tmp = Template(tmpl)
        return tmp.render(params).strip()

    @classmethod
    def __save(cls, code_text, filename=None):
        with open(filename, 'w') as f:
            for p in code_text:
                f.write(p)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("configure-file")
    parser.add_argument("-config", type=str,
                        help="config file")

    args = parser.parse_args()
    dispatch_task = DispatchTasks(user_configure_path="/home/liangqian/Gauss/experiments/h95Mhl/train_user_config.yaml",
                                  system_configure_path="/home/liangqian/Gauss/configure_files/dispatch_system_config/dispatch_system_config.yaml")
    dispatch_task.run()
