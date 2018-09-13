from __future__ import print_function
__author__ = 'Xi Ouyang'
"""
Creat the activation layer.
"""

import os
import sys
import timeit

import theano.tensor as T

class Activation(object):
    """
    activation function
    """
    def __init__(self, input, activation):
        """

        :param input: symbolic video tensor of shape video_shape
        :param activation: sigmoid/tanh/relu
        :return: the same shape as input
        """
        assert activation in ('sigmoid', 'tanh', 'relu')
        self.input = input

        if activation == 'sigmoid':
            self.output = T.nnet.sigmoid(input)
        if activation == 'tanh':
            self.output = T.tanh(input)
        if activation == 'relu':
            self.output = T.nnet.relu(input)

        # keep track of model input
        self.input = input

