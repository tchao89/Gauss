# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gauss import Component

class Model(Component):
    """model class packages real model for execution 
    """
    def __init__(self,
                 name: str,
                 train_flag: bool = True,
                 metric_name: str,
                 config: Dict[str, Any] = {}):
        self._metric_name = metric_name
        self._config = config
        super(Model, self).__init__(
            name = name,
            train_flag = train_flag
        )
    @abc.abstractmethod
    def model_save(self):
        pass

  
