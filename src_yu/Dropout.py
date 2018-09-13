from __future__ import print_function
__author__ = 'Xi Ouyang'
"""
dropout layer.
"""

import theano.tensor as tensor


def Dropout(input, use_noise, trng):
    output = tensor.switch(use_noise,
                         (input *
                          trng.binomial(input.shape,
                                        p=0.5, n=1,
                                        dtype=input.dtype)),
                           input * 0.5)
    return output