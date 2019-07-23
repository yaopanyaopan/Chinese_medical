# -*-coding:utf-8 -*-
__author__ = '$'

import tensorflow as tf
import numpy as np

class Model(object):

    def build_graph(self,dimension, lamda):

        self.dimension = dimension
        initializer = tf.contrib.layers.xavier_initializer()  # 用来在所有层中保持梯度大体相同

        self.x_train = tf.placeholder(tf.float32,shape=[None, 2*self.dimension])
        self.y_train = tf.placeholder(tf.float32,shape=[None, 1])

        self.theta0 = tf.get_variable(initializer =initializer,shape=[self.dimension,1],name='w0')
        self.theta1 = tf.get_variable(initializer =initializer,shape=[self.dimension,1],name='w1')

        self.bias = tf.Variable(tf.zeros([1]), name='bias')
        self.theta_new = tf.concat([self.theta0,self.theta1],0)

        self.y = 1 / (1 + tf.exp(-tf.matmul(self.x_train, self.theta_new)))

        self.loss = tf.reduce_mean(- self.y_train * tf.log(self.y) - (1 - self.y_train) * tf.log(1 - self.y))\
               + tf.contrib.layers.l1_regularizer(lamda)(self.theta0) + tf.contrib.layers.l1_regularizer(lamda)(self.theta1)


        self.train = tf.train.FtrlOptimizer(learning_rate=0.01,l1_regularization_strength=50).minimize(self.loss)

        self.acc =tf.equal(self.y,self.y_train)






















