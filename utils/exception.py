# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: Lab
class PipeLineLogicError(Exception):
    """
    This exception will activation when preprocessing
    route is not compatible with core route.
    """
    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return self.message


class NoResultReturnException(Exception):
    """
    This exception will activate when no result
    returns in a graph.
    """
    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return self.message
