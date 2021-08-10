# -*- coding: utf-8 -*-
#    
def tf_global_config(intra_threads, inter_threads):
    import tensorflow as tf
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=intra_threads,
        inter_op_parallelism_threads=inter_threads,
    )
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()