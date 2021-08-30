# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
class PipeLineLogicError(Exception):
    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return self.message


class NoResultReturnException(Exception):
    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return self.message
