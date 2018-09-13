from __future__ import print_function
__author__ = 'Xi Ouyang'
"""
Creat the max-pooling layer.
"""

import os
import sys
import timeit

from theano.tensor.signal import downsample

class Poollayer(object):
    """
    Max-pooling layer
    """
    def __init__(self, input, poolsize):
        """
        :param input: symbolic video tensor of shape video_shape
        :param poolsize: max-pooling size
        :return: (batch, frame number of output,
                   number of feature maps (channels), frame height of video, frame width of video )
        """

        self.input = input

        pooled_out = downsample.max_pool_2d(
            input = input,
            ds = poolsize,
            ignore_border = True
        )

        self.output = pooled_out

        # keep track of model input
        self.input = input



