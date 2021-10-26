# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, Citic-Lab. All rights reserved.
# Authors: citic-lab

from core.tfdnn.utils.loggers import EarlyStopLogger 

class Earlystop(object):
    # TODO: extend more metrics to be monitored.
    """stop training when loss has stopped decreasing in global.
    
    Currently, just monitoring loss score of runing model. A 'trainer.run()'
    training loop will check at end of each epoch whether the loss decreased
    or not in global trend. Considering 'delta' and 'patience' if applied.
    Once decreasing trend dismiss, 'flag' will set to False to terminate training."""

    def __init__(self, patience=3, delta=0.001):
        """
        :param patience: number of epochs with no decreasing after 
            which trainign will be stopped.
        :param delta: minimum change in loss to qualify as a decreasing.
        """
        self._patience = patience
        self._delta = delta

        self.counter = 0
        self.min_loss = None
        self._flag = False

        self._logger = EarlyStopLogger()

    def __call__(self, epoch, loss):
        if self.min_loss is None:
            self.min_loss = loss
        if loss > self.min_loss+self._delta:
            self.counter += 1
            if self.counter > self._patience:
                self._flag = True
                self._logger.log_info(epoch, loss)
        else:
            self.min_loss = loss
            self.counter = 0

    @property
    def flag(self):
        """False for stop training, vice verse."""
        return self._flag