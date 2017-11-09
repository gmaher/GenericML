import numpy as np
import tensorflow as tf
def linear(x,output_size,scope='linear',reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        input_shape  = x.get_shape().as_list()
        output_shape = [input_shape[1],output_size]

        std = np.sqrt(2.0/input_shape[1])

        W = tf.get_variable('W', shape=output_shape,
            initializer=tf.random_normal_initializer(0.0, std))

        b = tf.get_variable("b", shape=output_shape[1],
            initializer=tf.random_normal_initializer(0.0, std))

        h = tf.matmul(x,W)+b
        return h

def fullyConnected(x,output_size,activation=tf.identity,scope='FC',reuse=False):
    """activation is tensorflow function, e.g. tf.sigmoid"""
    with tf.variable_scope(scope,reuse=reuse):
        h = linear(x,output_size,reuse=reuse)
        a = activation(h)
        return a

def DNN(x,output_size,hidden_sizes,activation,scope="DNN",reuse=False):
    o = x
    with tf.variable_scope(scope,reuse=reuse):
        for i,s in enumerate(hidden_sizes):
            fc_scope = 'FC_'+str(i)
            o = fullyConnected(o,s,activation,scope=fc_scope,reuse=reuse)

        fc_scope = 'FC_out_linear'
        o = fullyConnected(o,output_size,scope=scope,reuse=reuse)
        return o

def MSE(y,yhat):
    return tf.reduce_mean(tf.square(y-yhat))
