"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import os
import shutil

from utils.base import mkdir


def copy_folder(source_path, target_path):
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
            copy_folder(path_read_new, path_write_new)
        else:
            # 将文件path_read_new复制到path_write_new
            shutil.copyfile(path_read_new, path_write_new)
