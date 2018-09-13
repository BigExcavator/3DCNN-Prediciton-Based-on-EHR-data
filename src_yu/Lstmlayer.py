from __future__ import print_function
__author__ = 'Xi Ouyang'
"""
build LSTM model.
"""

import numpy
import theano
from theano import config
import theano.tensor as tensor


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def normal_weight(rng, n_in, n_out):
    bound = numpy.sqrt(2.0 / n_in)
    W = numpy.asarray(
        rng.normal(loc=0, scale=bound, size=(n_in, n_out)),
                   dtype=theano.config.floatX
                   )
    return W

class Lstmlayer(object):
    """
    lstm layer.
    """
    def __init__(self, rng, input, input_shape, hidden_num):
        """
        :param rng:
        :param input:
        :param hidden_num:
        :return:
        """
        self.input = input
        #print(input.shape[2])
        #initialize weights (W for h, U for X)
        if input_shape[2] == hidden_num:
            U_values = numpy.concatenate([ortho_weight(hidden_num),
                                   ortho_weight(hidden_num),
                                   ortho_weight(hidden_num),
                                   ortho_weight(hidden_num)], axis=1)
            self.U = theano.shared(U_values, borrow=True)
        else:
            U_values = numpy.concatenate([normal_weight(rng, input_shape[2], hidden_num),
                                        normal_weight(rng, input_shape[2], hidden_num),
                                        normal_weight(rng, input_shape[2], hidden_num),
                                        normal_weight(rng, input_shape[2], hidden_num)], axis=1)
            self.U = theano.shared(U_values, borrow=True)
        W_values = numpy.concatenate([ortho_weight(hidden_num),
                                      ortho_weight(hidden_num),
                                      ortho_weight(hidden_num),
                                      ortho_weight(hidden_num)], axis=1)
        self.W = theano.shared(W_values, borrow=True)
        b_values = numpy.zeros((4 * hidden_num,), dtype=theano.config.floatX)
        self.b = theano.shared(b_values, borrow=True)

        nsteps = input.shape[0]
        if input.ndim == 3:
            n_samples = input.shape[1]
        else:
            n_samples = 1

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(x_, h_, c_):
            preact = tensor.dot(h_, self.W)
            preact += x_

            i = tensor.nnet.sigmoid(_slice(preact, 0, hidden_num))
            f = tensor.nnet.sigmoid(_slice(preact, 1, hidden_num))
            o = tensor.nnet.sigmoid(_slice(preact, 2, hidden_num))
            c = tensor.tanh(_slice(preact, 3, hidden_num))

            c = f * c_ + i * c

            h = o * tensor.tanh(c)

            return h, c

        input = (tensor.dot(input, self.U) + self.b)

        dim_proj = hidden_num
        rval, updates = theano.scan(_step,
                                sequences=[input],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name='lstm_layers',
                                n_steps=nsteps)
        self.output = rval[0]
        self.params = [self.W, self.U, self.b]
        # keep track of model input
        self.input = input
