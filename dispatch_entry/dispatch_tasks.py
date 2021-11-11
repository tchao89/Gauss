"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: Lab
"""
import argparse
import os.path
import itertools

from jinja2 import Template

from utils.yaml_exec import yaml_read
from utils.yaml_exec import yaml_write
from utils.constant_values import ConstantValues
from utils.bunch import Bunch


class DispatchTasks:
    def __init__(self,
                 user_configure_path: str = None,
                 system_configure_path: str = None):
        self.__udf_tmpl_fun = """
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
        self.__auto_tmpl_fun = """
# -*- coding: utf-8 -*-

# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
from pipeline.dispatch_pipeline.dispatch_auto_modeling_graph import AutoModelingGraph

def {{fun}}(name="udf", user_configure=None, system_configure=None):
    if user_configure is None:
        user_configure = {{user_configure}}

    if system_configure is None:
        system_configure = {{system_configure}}

    model_graph = AutoModelingGraph(name=name,
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
        docker_num = 0

        self.__main_id = os.path.split(user_configure[ConstantValues.work_root])[1]
        self.__work_space = os.path.split(user_configure[ConstantValues.work_root])[0]

        generated_ids = []
        data_clear_flag = user_configure.data_clear_flag
        feature_generator_flag = user_configure.feature_generator_flag
        unsupervised_feature_selector_flag = user_configure.unsupervised_feature_selector_flag
        supervised_feature_selector_flag = user_configure.supervised_feature_selector_flag
        supervised_selector_model_names = system_configure.supervised_selector_model_names
        opt_model_names = system_configure.opt_model_names
        model_zoo = user_configure.model_zoo

        routes = self.__auto_generate_route(data_clear_flag=data_clear_flag,
                                            feature_generator_flag=feature_generator_flag,
                                            unsupervised_feature_selector_flag=unsupervised_feature_selector_flag,
                                            supervised_feature_selector_flag=supervised_feature_selector_flag,
                                            supervised_selector_model_names=supervised_selector_model_names,
                                            opt_model_names=opt_model_names,
                                            model_zoo=model_zoo)
        for params in routes:
            data_clear_flag, feature_generator_flag, unsupervised_feature_selector_flag, \
                supervised_feature_selector_flag, supervised_selector_model_names, \
                opt_model_names, model_name = params

            if data_clear_flag is False and feature_generator_flag is True:
                continue

            if feature_generator_flag is True and unsupervised_feature_selector_flag is False:
                continue

            if feature_generator_flag is True and supervised_feature_selector_flag is False:
                continue

            folder_name = "_".join([str(data_clear_flag),
                                    str(feature_generator_flag),
                                    str(unsupervised_feature_selector_flag),
                                    str(supervised_feature_selector_flag),
                                    str(supervised_selector_model_names),
                                    opt_model_names,
                                    model_name])

            generated_id = self.__main_id + "-" + folder_name
            generated_ids.append(generated_id)

            temp_work_root = os.path.join(self.__work_space, generated_id)
            user_configure[ConstantValues.work_root] = temp_work_root
            user_configure[ConstantValues.data_clear_flag] = data_clear_flag
            user_configure[ConstantValues.feature_generator_flag] = feature_generator_flag
            user_configure[ConstantValues.unsupervised_feature_selector_flag] = unsupervised_feature_selector_flag
            user_configure[ConstantValues.supervised_feature_selector_flag] = supervised_feature_selector_flag
            system_configure[ConstantValues.supervised_selector_model_names] = supervised_selector_model_names
            system_configure[ConstantValues.opt_model_names] = opt_model_names
            system_configure[ConstantValues.model_name] = model_name

            train_func = self.__render(self.__auto_tmpl_fun,
                                       fun="main_train",
                                       user_configure=user_configure,
                                       system_configure=system_configure)

            self.__save(code_text=train_func, filename="./dispatch_methods/" + generated_id + ".py")
            docker_num += 1

        if docker_num <= 0:
            raise ValueError("Value: docker_num is {}, but it should be more than zero.".format(docker_num))

        if docker_num <= 0:
            raise ValueError("Value: docker_num is {}, but it should be more than zero.".format(docker_num))

        self.__dispatch_configure.docker_num = docker_num
        self.__dispatch_configure.generated_ids = generated_ids
        self.__dispatch_configure.main_id = self.__main_id
        self.__dispatch_configure.work_space = self.__work_space

    def __udf_dispatch(self, user_configure, system_configure):
        docker_num = 0

        self.__main_id = os.path.split(user_configure[ConstantValues.work_root])[1]
        self.__work_space = os.path.split(user_configure[ConstantValues.work_root])[0]

        generated_ids = []
        supervised_selector_model_names = system_configure.supervised_selector_model_names
        opt_model_names = system_configure.opt_model_names
        model_zoo = user_configure.model_zoo

        routes = self.__udf_generate_route(supervised_selector_model_names=supervised_selector_model_names,
                                            opt_model_names=opt_model_names,
                                            model_zoo=model_zoo)
        for params in routes:
            supervised_selector_model_names, opt_model_names, model_name = params
            folder_name = "_".join([supervised_selector_model_names,
                                    opt_model_names,
                                    model_name])
            generated_id = self.__main_id + "-" + folder_name
            generated_ids.append(generated_id)

            temp_work_root = os.path.join(self.__work_space, generated_id)
            user_configure[ConstantValues.work_root] = temp_work_root

            train_func = self.__render(self.__udf_tmpl_fun,
                                       fun="main_train",
                                       user_configure=user_configure,
                                       system_configure=system_configure)

            self.__save(code_text=train_func, filename="./dispatch_methods/" + generated_id + ".py")
            docker_num += 1

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

    @classmethod
    def __auto_generate_route(cls, **params):
        supervised_selector_model_names = params[ConstantValues.supervised_selector_model_names]
        if not isinstance(supervised_selector_model_names, list):
            raise TypeError(
                "Value: supervised selector model names should be type of list, "
                "but get {} instead.".format(
                    supervised_selector_model_names)
            )

        opt_model_names = params[ConstantValues.opt_model_names]
        if not isinstance(opt_model_names, list):
            raise TypeError(
                "Value: opt model names should be type of list, "
                "but get {} instead.".format(
                    opt_model_names)
            )

        data_clear_flag = params[ConstantValues.data_clear_flag]
        if not isinstance(data_clear_flag, list):
            raise TypeError(
                "Value: data clear flag should be type of list, "
                "but get {} instead.".format(
                    data_clear_flag)
            )

        feature_generator_flag = params[ConstantValues.feature_generator_flag]
        if not isinstance(feature_generator_flag, list):
            raise TypeError(
                "Value: feature generator flag should be type of list, "
                "but get {} instead.".format(
                    feature_generator_flag)
            )

        unsupervised_feature_selector_flag = params[ConstantValues.unsupervised_feature_selector_flag]
        if not isinstance(unsupervised_feature_selector_flag, list):
            raise TypeError(
                "Value: unsupervised feature selector flag should be type of list, "
                "but get {} instead.".format(
                    unsupervised_feature_selector_flag)
            )

        supervised_feature_selector_flag = params[ConstantValues.supervised_feature_selector_flag]
        if not isinstance(supervised_feature_selector_flag, list):
            raise TypeError(
                "Value: supervised feature selector flag should be type of list, "
                "but get {} instead.".format(
                    supervised_feature_selector_flag)
            )

        model_zoo = params["model_zoo"]
        if not isinstance(model_zoo, list):
            raise TypeError(
                "Value: model zoo should be type of list, "
                "but get {} instead.".format(
                    model_zoo)
            )

        routes = itertools.product(
            data_clear_flag,
            feature_generator_flag,
            unsupervised_feature_selector_flag,
            supervised_feature_selector_flag,
            supervised_selector_model_names,
            opt_model_names,
            model_zoo)
        return routes

    @classmethod
    def __udf_generate_route(cls, **params):
        supervised_selector_model_names = params[ConstantValues.supervised_selector_model_names]
        if not isinstance(supervised_selector_model_names, list):
            raise TypeError(
                "Value: supervised selector model names should be type of list, "
                "but get {} instead.".format(
                    supervised_selector_model_names)
            )

        opt_model_names = params[ConstantValues.opt_model_names]
        if not isinstance(opt_model_names, list):
            raise TypeError(
                "Value: opt model names should be type of list, "
                "but get {} instead.".format(
                    opt_model_names)
            )

        model_zoo = params["model_zoo"]
        if not isinstance(model_zoo, list):
            raise TypeError(
                "Value: model zoo should be type of list, "
                "but get {} instead.".format(
                    model_zoo)
            )

        routes = itertools.product(
            supervised_selector_model_names,
            opt_model_names,
            model_zoo)
        return routes


if __name__ == "__main__":
    parser = argparse.ArgumentParser("configure-file")
    parser.add_argument("-config", type=str,
                        help="config file")

    args = parser.parse_args()
    dispatch_task = DispatchTasks(user_configure_path="/home/liangqian/Gauss/experiments/j97qq5/train_user_config.yaml",
                                  system_configure_path="/home/liangqian/Gauss/configure_files/system_config/system_config.yaml")
    dispatch_task.run()
