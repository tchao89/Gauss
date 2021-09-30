# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from core.tfdnn.networks.mlp_network import (
    MlpRegNetwork,
    MlpClsNetwork
)


class NetworkFactory():

    CLS = "classification"
    REG = "regression"

    @staticmethod
    def get_network(task_name):
        if task_name == NetworkFactory.CLS:
            return MlpClsNetwork
        elif task_name == NetworkFactory.REG:
            return MlpRegNetwork
        else:
            raise NotImplementedError(
                "Current task is not supported yet."
            )