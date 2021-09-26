"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import os

from utils.yaml_exec import yaml_read, yaml_write


def reconstruct_folder(folder, init_prefix=None):
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
                path_replace(source_dict=configure_dict,
                             init_prefix=init_prefix,
                             prefix=prefix)

                yaml_write(yaml_dict=configure_dict, yaml_file=file_path)


def path_replace(source_dict, init_prefix=None, prefix=None):
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
                path_replace(
                    source_dict=source_dict[key],
                    init_prefix=init_prefix,
                    prefix=prefix
                )
