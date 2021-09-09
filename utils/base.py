"""
-*- coding: utf-8 -*-

Copyright (c) 2020, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import os
import psutil


def mkdir(path: str):
    try:
        os.mkdir(path=path)
    except FileNotFoundError:
        os.system("mkdir -p " + path)


def get_current_memory_gb() -> dict:

    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    #  + info.swap / 1024. / 1024. / 1024. + info.pss / 1024. / 1024. / 1024.
    memory_usage = info.uss / 1024. / 1024. / 1024.

    return {"memory_usage": memory_usage, "pid": pid}
