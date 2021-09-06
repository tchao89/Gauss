# -*- coding: utf-8 -*-
#    
from typing import Any

def tf_global_config(intra_threads, inter_threads):
    import tensorflow as tf
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=intra_threads,
        inter_op_parallelism_threads=inter_threads,
    )
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()


class CONST:
    
    def __call__(self, name, value):
        self.__setattr__(name, value)
        return self.__dict__[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            raise self.ConstError("can't rebind const {name}".format(
                name=name
            ))
        self.__dict__[name] = value