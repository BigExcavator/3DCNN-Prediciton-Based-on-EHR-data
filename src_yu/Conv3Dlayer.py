from __future__ import print_function
__author__ = 'Xi Ouyang'
"""
Creat the 3D convolutional layer.
"""

import os
import sys
import timeit

import numpy

import theano
from theano.tensor.nnet.conv3d2d import conv3d


class Conv3Dlayer(object):
    """
    3D convoluiton layer
    """
    def __init__(self, rng, input, video_shape, filter_shape):
        """
        :param rng: a random number generator used to initialize weights
        :param input: symbolic video tensor of shape video_shape
        :param video_shape: (batch, frame number of video (input temporal length),
                              number of feature maps (channels), frame height of video, frame width of video )
        :param filter_shape: (number of output feature maps, filter temporal length,
                               number of input feature maps (channels), filter height, filter width)
        :return: (batch, frame number of output,
                   number of feature maps (channels), frame height of video, frame width of video )
        """
        assert video_shape[2] == filter_shape[2]
        self.input = input

        #fan_in is the number of units in the (i ? 1)-th layer
        fan_in = numpy.prod(filter_shape[1:])
        #fan_out is the number of units in the i-th layer
        fan_out = numpy.prod(filter_shape[0:2]) * numpy.prod(filter_shape[3:])
        #initialize weights (if 'tanh' is the activation function)
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        
        #if flag == 0:
        self.W = theano.shared(
            numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX
          ),
          borrow=True
        )
        #else :
        #    self.W = W
        """
        #initialize weights (if 'sigmoid' is the action function)
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low= -4 * W_bound, high= 4 * W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        """
        """
        #initialize weights (another way in uniform distribution)
        W_bound = numpy.sqrt(1.0 / fan_in)
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        """
        """
        #initialize weights (normal distribution)
        W_bound = numpy.sqrt(2.0 / fan_in)
        self.W = theano.shared(
            numpy.asarray(
                rng.normal(rng.normal(loc=0, scale=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        """
        #initialize bias, the bias is a 1D tensor -- one bias per output feature map
        
        #if flag == 0:
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
           
        #else: 
        #    self.b = b   
        
          

        # convolve input feature maps with filters
        conv_out = conv3d(
            signals = input, #(batch, time, in channel, height, width)
            filters = self.W, #(out channel,time,in channel, height, width)
            signals_shape = video_shape,
            filters_shape = filter_shape,
            border_mode = 'valid'
        )

        #add bias to every feature map
        self.output = conv_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x')

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input





