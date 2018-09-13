from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class BatchNormalization(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, input_shape):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.gamma = theano.shared(
            value=numpy.ones(input_shape, dtype=theano.config.floatX),
            name='gamma',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.beta = theano.shared(
            value=numpy.zeros(
                input_shape,
                dtype=theano.config.floatX
            ),
            name='beta',
            borrow=True
        )

        self.input = input

        self.mean = T.mean(self.input,axis=0)

        self.std = T.std(self.input,axis=0)

        self.output = T.nnet.bn.batch_normalization(self.input, self.gamma, self.beta, self.mean, self.std)


