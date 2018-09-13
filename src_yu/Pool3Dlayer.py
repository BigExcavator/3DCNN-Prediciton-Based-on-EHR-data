import numpy
import theano
from theano.tensor.signal.downsample import *


class MaxPool3DLayer(object):
    """3D Layer of a convolutional network """

    def __init__(self, input, ds, ignore_border=False):
        """
        Allocate a layer for 3d max-pooling.

        The layer takes as input a 5-D tensor. It downscales the input image by
        the specified factor, by keeping only the maximum value of non-overlapping
        patches of size (ds[0],ds[1])
        :type input: 5-D Theano tensor of input 3D images.
        :param input: input images. Max pooling will be done over the 2 last
            dimensions (x, y), and the second dimension (z).
        :type ds: int
        :param ds: factor by which to downscale (same on all 3 dimensions).
            2 will halve the image in each dimension.
        :type ignore_border: bool
        :param ignore_border: When True, (5,5,5) input with ds=2
            will generate a (2,2,2) output. (3,3,3) otherwise.
        """

        # max_pool_2d X and Z
        temp_output = max_pool_2d(input=input.dimshuffle(0, 4, 2, 3, 1),
                                  ds=(ds[1], ds[0]),
                                  ignore_border=ignore_border,)
        # temp_output_shape = DownsampleFactorMax.out_shape(imgshape=(input_shape[0], input_shape[4], input_shape[2], input_shape[3], input_shape[1]),
                                                          # ds=(ds[1], ds[0]),
                                                          # ignore_border=ignore_border)

        # max_pool_2d X and Y (with X constant)
        output = max_pool_2d(input=temp_output.dimshuffle(0, 4, 2, 3, 1),
                             ds=(1, ds[2]),
                             ignore_border=ignore_border)
        # output_shape = DownsampleFactorMax.out_shape(imgshape=(temp_output_shape[0], temp_output_shape[4], temp_output_shape[2], temp_output_shape[3], temp_output_shape[1]),
                                                     # ds=(1, ds[2]),
                                                     # ignore_border=ignore_border)

        self.input = input
        self.output = output

        # self.input_shape = input_shape
        # self.output_shape = output_shape
