# -*- coding: utf-8 -*- 

from __future__ import print_function
__author__ = 'Ruohan Sun, lei li'

"""
train our model
"""


import os
import sys
import time
import timeit

import numpy
import random

import h5py
import cPickle
import argparse
import six.moves.cPickle as pickle
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

from src_yu.logistic_sgd import LogisticRegression
from src_yu.mlp import HiddenLayer
from src_yu.Activation import Activation
from src_yu.Conv3Dlayer2 import Conv3Dlayer
from src_yu.Poollayer import Poollayer
from src_yu.Conv2Dlayer import Conv2DLayer
from src_yu.Dropout import Dropout
from src_yu.Pool3Dlayer import MaxPool3DLayer
from src_yu.Optimizer import Optimizer
from src_yu.batch_normalization import BatchNormalization

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset , or just data name such as UCF50
    :type net: string
    :param net: 3dcnn or lstm
    '''
    
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset + "_ou.h5"
        )
        if os.path.isfile(new_path):
            dataset = new_path
    
    #############
    # LOAD DATA #
    #############

    print('... loading data')
    if os.path.isfile( dataset ):
        # load data 
        file=h5py.File(dataset,'r')
        print ('-------read train-----------')
        train_x1 = file['train_x1'][:]
        print ( 'train_x1' + str(train_x1.shape) )
        train_y1 = file['train_y1'][:]
        print ( 'train_y1' + str(train_y1.shape) )
        
        train_x2 = file['train_x2'][:]
        print ( 'train_x2' + str(train_x2.shape) )
        train_y2 = file['train_y2'][:]
        print ( 'train_y2' + str(train_y2.shape) )
        
        train_x3 = file['train_x3'][:]
        print ( 'train_x3' + str(train_x3.shape) )
        train_y3 = file['train_y3'][:]
        print ( 'train_y3' + str(train_y3.shape) )
        
        train_x4 = file['train_x4'][:]
        print ( 'train_x4' + str(train_x4.shape) )
        train_y4 = file['train_y4'][:]
        print ( 'train_y4' + str(train_y4.shape) )
        
        
        
        print ( '-------read valid-----------' )
        valid_x1 = file['valid_x1'][:]
        print ( 'valid_x1' + str(valid_x1.shape) )
        valid_y1 = file['valid_y1'][:]
        print ( 'valid_y1' + str(valid_y1.shape) )
        
        valid_x2 = file['valid_x2'][:]
        print ( 'valid_x2' + str(valid_x2.shape) )
        valid_y2 = file['valid_y2'][:]
        print ( 'valid_y2' + str(valid_y2.shape) )
        
        
        valid_x3 = file['valid_x3'][:]
        print ( 'valid_x3' + str(valid_x3.shape) )
        valid_y3 = file['valid_y3'][:]
        print ( 'valid_y3' + str(valid_y3.shape) )
        
        valid_x4 = file['valid_x4'][:]
        print ( 'valid_x4' + str(valid_x4.shape) )
        valid_y4 = file['valid_y4'][:]
        print ( 'valid_y4' + str(valid_y4.shape) )
        
        
        
        
        
        
        print ( '-------read test-----------' )
        test_x1 = file['test_x1'][:]
        print ( 'test_x1' + str(test_x1.shape) )
        test_y1 = file['test_y1'][:]
        print ( 'test_y1' + str(test_y1.shape) )
        
        test_x2 = file['test_x2'][:]
        print ( 'test_x2' + str(test_x2.shape) )
        test_y2 = file['test_y2'][:]
        print ( 'test_y2' + str(test_y2.shape) )
        
        test_x3 = file['test_x3'][:]
        print ( 'test_x3' + str(test_x3.shape) )
        test_y3= file['test_y3'][:]
        print ( 'test_y3' + str(test_y3.shape) )
        
        test_x4 = file['test_x4'][:]
        print ( 'test_x4' + str(test_x4.shape) )
        test_y4 = file['test_y4'][:]
        print ( 'test_y4' + str(test_y4.shape) )
        file.close()
    
        
    print ('\nloaded data train_x[1, 2, 0, 1] ' )
    

    
    rval = [(train_x1.astype('float32'), train_y1.astype('float32')), (valid_x1.astype('float32'), valid_y1.astype('float32')), (test_x1.astype('float32'), test_y1.astype('float32')),
            (train_x2.astype('float32'), train_y2.astype('float32')), (valid_x2.astype('float32'), valid_y2.astype('float32')), (test_x2.astype('float32'), test_y2.astype('float32')),
            (train_x3.astype('float32'), train_y3.astype('float32')), (valid_x3.astype('float32'), valid_y3.astype('float32')), (test_x3.astype('float32'), test_y3.astype('float32')),
            (train_x4.astype('float32'), train_y4.astype('float32')), (valid_x4.astype('float32'), valid_y4.astype('float32')), (test_x4.astype('float32'), test_y4.astype('float32'))]
    #file = open(os.path.join(os.path.split(__file__)[0], "data",'W.pkl'),'rb')
    #words = cPickle.load(file)
    #words = words.astype(theano.config.floatX)
    #return rval, words
    return rval

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params   

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)
    
def init_params(options):
    """
    Global parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    #params = get_layer(options['encoder'])[0](options, params, prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['n_fc'],
                                            options['n_label']).astype(theano.config.floatX)
    params['b'] = numpy.zeros((options['n_label'],)).astype(theano.config.floatX)

    return params


#def init_tparams(params, words):
#    tparams = OrderedDict()
#    for kk, pp in params.items():
#        tparams[kk] = theano.shared(params[kk], name=kk)
#    tparams['words'] = theano.shared(words, borrow=True)
#    return tparams
def init_tparams(params):
    tparams = OrderedDict()
    for kk,pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    #tparams['words'] = theano.shared(words, borrow=True)
    return tparams

    
def load_params(path, tparams):
    pp = numpy.load(path)
    for kk, vv in tparams.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        tparams[kk].set_value(pp[kk])
    
def get_minibatches_idx(num, n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    number = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
        number.append(num)

    return zip(number, range(len(minibatches)), minibatches)
    
#def L1_norm(param):
    #return abs(param).sum()

def L2_norm(param):
    return (param ** 2).sum()
    
def build_model(tparams, options):
     ######################
    # BUILD ACTUAL MODEL #
    ######################
    nkerns = options['nkerns']
    n_fc = options['n_fc']
    n_in = options['n_label']
    batch_size = options['batch_size']
    rng = numpy.random.RandomState(23455)
    rng_marix = numpy.random.RandomState(42)
    
    index = T.lscalar()  # index to a [mini]batch

    #itensor3 = T.TensorType('float32', (False,)*3)
    #x = itensor3('x')   # the data is presented as rasterized images
    itensor4 = T.TensorType('float32', (False,)*4)
    x = itensor4('x')
    y = T.fvector('y')  # the labels are presented as 1D vector of   
    z = T.matrix('z')
   
    #words_input = x.reshape((x.shape[0]*x.shape[1]*x.shape[2],))
    print(x.type,y.type)

    #def words_vec(i):
    #    return ifelse(T.ge(words_input[i], 0), tparams['words'][words_input[i]], EOS)
        #return tparams['words'][words_input[i]]

    #results, updates = theano.scan(words_vec, sequences=[T.arange(words_input.shape[0])])

    network_input = x.reshape((x.shape[0], x.shape[1], 1, x.shape[2], x.shape[3]))

    conv1 = Conv3Dlayer(
            rng,
            input=network_input,
            filter_shape=(nkerns[0],5,1,2,50)
        )
    relu1 = Activation(conv1.output, 'relu')
    
    pool1 = ifelse(T.eq(x.shape[1], 98),
            MaxPool3DLayer(input = relu1.output, ds=(2, 2, 1)).output,
            ifelse(T.eq(x.shape[1], 142),
                MaxPool3DLayer(input = relu1.output, ds=(3, 2, 1)).output,
                ifelse(T.eq(x.shape[1], 186),
                    MaxPool3DLayer(input = relu1.output, ds=(3, 2, 1)).output,
                    MaxPool3DLayer(input = relu1.output, ds=(4, 2, 1)).output,
                )
            )
        )
    
    conv2 = Conv3Dlayer(
            rng,
            input=pool1,
            filter_shape=(nkerns[1], 2, nkerns[0], 1, 1)
        )
    relu2 = Activation(conv2.output, 'relu')
    
    pool2 = ifelse(T.eq(x.shape[1], 98),
            MaxPool3DLayer(input = relu2.output, ds=(4, 1, 1)).output,
            ifelse(T.eq(x.shape[1], 142),
                MaxPool3DLayer(input = relu2.output, ds=(4, 1, 1)).output,
                ifelse(T.eq(x.shape[1], 186),
                    MaxPool3DLayer(input = relu2.output, ds=(5, 1, 1)).output,
                    MaxPool3DLayer(input = relu2.output, ds=(5, 1, 1)).output,
                )
            )
        )
         
    '''       
    conv3 = Conv3Dlayer(
            rng,
            input=pool2,
            filter_shape=(nkerns[2], 3, nkerns[1], 1, 1)
        )
    relu3 = Activation(conv3.output, 'relu')
    
    pool3 = ifelse(T.eq(x.shape[1], 54),
            MaxPool3DLayer(input = relu3.output, ds=(1, 1, 1)).output,
            ifelse(T.eq(x.shape[1], 65),
                MaxPool3DLayer(input = relu3.output, ds=(3, 1, 1)).output,
                ifelse(T.eq(x.shape[1], 77),
                    MaxPool3DLayer(input = relu3.output, ds=(4, 1, 1)).output,
                    MaxPool3DLayer(input = relu3.output, ds=(1, 1, 1)).output,
                )
            )
        )
        
        
        
    conv4 = Conv3Dlayer(
            rng,
            input=pool3,
            filter_shape=(nkerns[3], 2, nkerns[2], 1, 1)
        )
    relu4 = Activation(conv4.output, 'relu')
    
    pool4 = ifelse(T.eq(x.shape[1], 54),
            MaxPool3DLayer(input = relu4.output, ds=(1, 1, 1)).output,
            ifelse(T.eq(x.shape[1], 65),
                MaxPool3DLayer(input = relu4.output, ds=(3, 1, 1)).output,
                ifelse(T.eq(x.shape[1], 77),
                    MaxPool3DLayer(input = relu4.output, ds=(2, 1, 1)).output,
                    MaxPool3DLayer(input = relu4.output, ds=(2, 1, 1)).output,
                )
            )
        )  
    '''  
    #spp
    #pool_spp_2 = MaxPool3DLayer(input = pool4, ds=(3, 1, 1)).output
    #pool_spp_1 = MaxPool3DLayer(input = pool4, ds=(6, 1, 1)).output
    pool_spp_2 = MaxPool3DLayer(input = pool2, ds=(4, 1, 1)).output
    pool_spp_1 = MaxPool3DLayer(input = pool2, ds=(6, 1, 1)).output
    
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    # conv2_input = pool2.reshape((batch_size, 54, 2, 10))
    # conv2 = Conv2DLayer(
        # rng,
        # input= conv2_input,
        # video_shape=(batch_size, 54, 2, 10),
        # filter_shape=(nkerns[1], 54, 2, 10)
    # )
    # tanh3 = Activation(conv2.output, 'tanh')
    # fc1_input = tanh3.output.flatten(2)
    
    
    z = T.concatenate([pool2.flatten(2), pool_spp_2.flatten(2), pool_spp_1.flatten(2) ],axis = 1)
 
    
    fc1_input = z
    

    fc1 = HiddenLayer(
        rng,
        input=fc1_input,
        #n_in=nkerns[3]*9*1*1,
        n_in=nkerns[1]*17*1*1,
        n_out=n_fc,
        activation=T.nnet.relu
    )
    # construct a fully-connected sigmoidal layer
    
    # conv2 = Conv3Dlayer(
            # rng,
            # input=pool2,
            # video_shape=(batch_size,9, nkerns[0], 2, 1),
            # filter_shape=(nkerns[1], 1, nkerns[0], 2, 1)
        # )
    
    # fc1_input = conv2.output.flatten(2)
    
    #produce use_noise for dropout
    
    use_noise = theano.shared(numpy.asarray (0., dtype = theano.config.floatX))
    
    #produce trng for dropout
    SEED = 123
    trng = RandomStreams(SEED)

    Droplayer = Dropout(fc1.output, use_noise, trng )
    #BN = BatchNormalization(
       #input = fc1.output, 
       #input_shape = (batch_size, n_fc)   
    #)
    
    # classify the values of the fully-connected sigmoidal layer
    pred = T.nnet.softmax(T.dot(Droplayer, tparams['U']) + tparams['b'])
    f_pred_prob = theano.function([x], pred, name='f_pred_prob')
    f_pred = theano.function([x], pred.argmax(axis=1), name='f_pred')

    # the cost we minimize during training is the NLL of the model
    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    y1=y[:].astype("int32")
    cost = -T.log(pred[T.arange(batch_size), y1] + off).mean()
    
    tparams['fc1_W'] = fc1.params[0]
    tparams['fc1_b'] = fc1.params[1]
    tparams['conv1_W'] = conv1.params[0]
    tparams['conv1_b'] = conv1.params[1]
    tparams['conv2_W'] = conv2.params[0]
    tparams['conv2_b'] = conv2.params[1]
    #tparams['conv3_W'] = conv3.params[0]
    #tparams['conv3_b'] = conv3.params[1]
    #tparams['conv4_W'] = conv4.params[0]
    #tparams['conv4_b'] = conv4.params[1]
    #tparams['bn_gamma'] = BN.gamma
    #tparams['bn_beta'] = BN.beta
    
    # for (params_name, params_value) in tparams.items():
        # if params_name.endswith("W") or params_name.endswith("U"):
            # L2 += L2_norm(params_value)
    # '''
    # L1 = L1_norm(tparams['U']) + L1_norm(tparams['fc1_W'])
    L2 = L2_norm(tparams['U']) + L2_norm(tparams['fc1_W'])    
    cost += options['L2_reg'] * L2

    return use_noise, x, y, f_pred_prob, f_pred, cost

def pred_error(f_pred, data_x, data_y, iterator, add_num=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    data_x_shape = 0
    for number, _, valid_index in iterator:
        if add_num:
            y = data_y[number][valid_index]
            x = data_x[number][valid_index]
        else:
            y = data_y[valid_index]
            x = data_x[valid_index]  
        preds = f_pred(x)
        valid_err += (preds == y).sum()
    if add_num:
        for i in range(len(data_x)):
            data_x_shape += data_x[i].shape[0]
        valid_err = 1. - numpy.asarray(valid_err, dtype=theano.config.floatX) / data_x_shape
    else:
        valid_err = 1. - numpy.asarray(valid_err, dtype=theano.config.floatX) / data_x.shape[0]

    return valid_err    

def evaluate_model( learning_rate=0.01, 
                    n_epochs=200, 
                    #dataset='stanfordSentimentTreebank',
                        dataset='data/h5/diabetes_cp_k3_new_98_142_186_246_p712.h5',
                    reload_model=None, # Path to a saved model we want to start from.
                    saveto='data/model/diabetes_cp_k3_new_model_1.npz',  # The best model will be saved there
                    validFreq=370,  # Compute the validation error after this number of update.
                    saveFreq=1110,  # Save the parameters after every saveFreq updates
                    dispFreq=10,  # Display to stdout the training progress every N updates
                    max_epochs=5000,  # The maximum number of epoch to run
                    patience=10,  # Number of epoch to wait before early stop if no progress
                    pat=3,  # Number of epoch to reduce learning_rate if no progress
                    k=3, 
                    nkerns=[30, 50, 100],
                    n_fc=50, #fully-connected layer 
                    n_label=2, #label                   
                    batch_size=10,
                    #valid_batch_size=10
                    L1_reg=0.001,
                    L2_reg=0.001
                    ):
    """
    test 3dcnn in random matrix
    :param learning_rate:
    :param n_epochs:
    :param nkerns: numbers of feature maps in all convolution layers
    :param batch_size:
    :return:
    """
    
    model_options = locals().copy()
    print("model options", model_options)
    
    #load data
    #datasets, words = load_data(dataset = dataset + '_spp_' + str(k))
    datasets= load_data(dataset = dataset)
    train_set_x = []
    train_set_y = []
    valid_set_x = []
    valid_set_y = []
    test_set_x = []
    test_set_y = []

    train_set_x1, train_set_y1 = datasets[0]
    valid_set_x1, valid_set_y1 = datasets[1]
    test_set_x1, test_set_y1 = datasets[2]
    
    train_set_x2, train_set_y2 = datasets[3]
    valid_set_x2, valid_set_y2 = datasets[4]
    test_set_x2, test_set_y2 = datasets[5]
    
    train_set_x3, train_set_y3 = datasets[6]
    valid_set_x3, valid_set_y3 = datasets[7]
    test_set_x3, test_set_y3 = datasets[8]
    
    train_set_x4, train_set_y4 = datasets[9]
    valid_set_x4, valid_set_y4 = datasets[10]
    test_set_x4, test_set_y4 = datasets[11]
    
    #train_set_x4 = train_set_x4[:, 0:32, :] 
    #valid_set_x4 = valid_set_x4[:, 0:32, :]
    #test_set_x4 = test_set_x4[:, 0:32, :]
    
    
    train_set_x.append(train_set_x1)
    train_set_x.append(train_set_x2)
    train_set_x.append(train_set_x3)
    train_set_x.append(train_set_x4)
    train_set_y.append(train_set_y1)
    train_set_y.append(train_set_y2)
    train_set_y.append(train_set_y3)
    train_set_y.append(train_set_y4)
    valid_set_x.append(valid_set_x1)
    valid_set_x.append(valid_set_x2)
    valid_set_x.append(valid_set_x3)
    valid_set_x.append(valid_set_x4)
    valid_set_y.append(valid_set_y1)
    valid_set_y.append(valid_set_y2)
    valid_set_y.append(valid_set_y3)
    valid_set_y.append(valid_set_y4)
    test_set_x.append(test_set_x1)
    test_set_x.append(test_set_x2)
    test_set_x.append(test_set_x3)
    test_set_x.append(test_set_x4)
    test_set_y.append(test_set_y1)
    test_set_y.append(test_set_y2)
    test_set_y.append(test_set_y3)
    test_set_y.append(test_set_y4)
    
    train_num1 = train_set_x1.shape[0]
    valid_num1 = valid_set_x1.shape[0]
    test_num1 = test_set_x1.shape[0]
    
    train_num2 = train_set_x2.shape[0]
    valid_num2 = valid_set_x2.shape[0]
    test_num2 = test_set_x2.shape[0]
    
    train_num3 = train_set_x3.shape[0]
    valid_num3 = valid_set_x3.shape[0]
    test_num3 = test_set_x3.shape[0]
    
    train_num4 = train_set_x4.shape[0]
    valid_num4 = valid_set_x4.shape[0]
    test_num4 = test_set_x4.shape[0]
    
    train_num = train_num1 + train_num2 + train_num3 + train_num4
    valid_num = valid_num1 + valid_num2 + valid_num3 + valid_num4
    test_num = test_num1 + test_num2 + test_num3 + test_num4
    # compute number of minibatches for training, validation and testing
    # n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    # compute number of minibatches for training, validation and testing
    n_train_batches1 = train_set_x1.shape[0] // batch_size
    n_valid_batches1 = valid_set_x1.shape[0] // batch_size
    n_test_batches1 = test_set_x1.shape[0] // batch_size  
    
    n_train_batches2 = train_set_x2.shape[0] // batch_size
    n_valid_batches2 = valid_set_x2.shape[0] // batch_size
    n_test_batches2 = test_set_x2.shape[0] // batch_size
    
    n_train_batches3 = train_set_x3.shape[0] // batch_size
    n_valid_batches3 = valid_set_x3.shape[0] // batch_size
    n_test_batches3 = test_set_x3.shape[0] // batch_size
    
    n_train_batches4 = train_set_x4.shape[0] // batch_size
    n_valid_batches4 = valid_set_x4.shape[0] // batch_size
    n_test_batches4 = test_set_x4.shape[0] // batch_size
                     
    params = init_params(model_options)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)
    #EOS = numpy.zeros(300, dtype=theano.conig.floatX)
    #EOS = numpy.zeros(50, dtype=theano.config.floatX)
    #EOS = theano.shared(EOS, borrow=True)
    
    print('... building the model')
    (use_noise, x, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)
    print ("========================building success============================")
    # after build model, get tparams
    if reload_model:
        load_params('nlp_spp_model.npz', tparams)
    
    f_cost = theano.function([x, y], cost, name='f_cost')

    grads = T.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, y], grads, name='f_grad')

    lr = T.scalar(name='lr')
    optimizer = Optimizer(tparams, grads, [x, y], cost, 'rmsprop', lr)
    
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    kf_valid1 = get_minibatches_idx(0, valid_num1, batch_size)
    kf_test1 = get_minibatches_idx(0, test_num1, batch_size)
    
    kf_valid2 = get_minibatches_idx(1, valid_num2, batch_size)
    kf_test2 = get_minibatches_idx(1, test_num2, batch_size)
    
    kf_valid3 = get_minibatches_idx(2, valid_num3, batch_size)
    kf_test3 = get_minibatches_idx(2, test_num3, batch_size)
    
    kf_valid4 = get_minibatches_idx(3, valid_num4, batch_size)
    kf_test4 = get_minibatches_idx(3, test_num4, batch_size)
    
    kf_valid = kf_valid1 + kf_valid2 + kf_valid3 + kf_valid4
    kf_test = kf_test1 + kf_test2 + kf_test3 + kf_test4
    
    print("x1 %d train examples" % train_num1)
    print("x1 %d valid examples" % valid_num1)
    print("x1 %d test examples" % test_num1)
    
    print("x2 %d train examples" % train_num2)
    print("x2 %d valid examples" % valid_num2)
    print("x2 %d test examples" % test_num2)
    
    print("x3 %d train examples" % train_num3)
    print("x3 %d valid examples" % valid_num3)
    print("x3 %d test examples" % test_num3)
    
    print("x4 %d train examples" % train_num4)
    print("x4 %d valid examples" % valid_num4)
    print("x4 %d test examples" % test_num4)
    
    print("%d train examples" % train_num)
    print("%d valid examples" % valid_num)
    print("%d test examples" % test_num)

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = train_num // batch_size
    if saveFreq == -1:
        saveFreq = train_num // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0
            # tidx = 0

            # Get new shuffled index for the training set.
            kf1 = get_minibatches_idx(0, train_num1, batch_size, shuffle=True)
            kf2 = get_minibatches_idx(1, train_num2, batch_size, shuffle=True)
            kf3 = get_minibatches_idx(2, train_num3, batch_size, shuffle=True)
            kf4 = get_minibatches_idx(3, train_num4, batch_size, shuffle=True)
            kf = kf1 + kf2 + kf3 + kf4
            random.shuffle(kf)
            #train_err1 = pred_error(f_pred, train_set_x1, train_set_y1, kf1)
            #valid_err1 = pred_error(f_pred, valid_set_x1, valid_set_y1, kf_valid1)
            #test_err1 = pred_error(f_pred, test_set_x1, test_set_y1, kf_test1)
            #train_err2 = pred_error(f_pred, train_set_x2, train_set_y2, kf2)
            #valid_err2 = pred_error(f_pred, valid_set_x2, valid_set_y2, kf_valid2)
            #test_err2 = pred_error(f_pred, test_set_x2, test_set_y2, kf_test2)
            #train_err3 = pred_error(f_pred, train_set_x3, train_set_y3, kf3)
            #valid_err3 = pred_error(f_pred, valid_set_x3, valid_set_y3, kf_valid3)
            #test_err3 = pred_error(f_pred, test_set_x3, test_set_y3, kf_test3)
            #train_err4 = pred_error(f_pred, train_set_x4, train_set_y4, kf4)
            #valid_err4 = pred_error(f_pred, valid_set_x4, valid_set_y4, kf_valid4)
            #test_err4 = pred_error(f_pred, test_set_x4, test_set_y4, kf_test4)
            #train_err = pred_error(f_pred, train_set_x, train_set_y, kf, add_num=True)
            #valid_err = pred_error(f_pred, valid_set_x, valid_set_y, kf_valid, add_num=True)
            #test_err = pred_error(f_pred, test_set_x, test_set_y, kf_test, add_num=True)
            
            #print( ('Train---------> ', train_err, 'Valid-------> ', valid_err,
            #               'Test--------> ', test_err) )

            for number, _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                y = train_set_y[number][train_index]
                x = train_set_x[number][train_index]
                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                n_samples += x.shape[0]
                # print('x_shape: ', x.shape)
                # print('y_shape: ', y.shape)
                cost = optimizer.update([x, y], learning_rate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err1 = pred_error(f_pred, train_set_x1, train_set_y1, kf1)
                    valid_err1 = pred_error(f_pred, valid_set_x1, valid_set_y1, kf_valid1)
                    test_err1 = pred_error(f_pred, test_set_x1, test_set_y1, kf_test1)
                    train_err2 = pred_error(f_pred, train_set_x2, train_set_y2, kf2)
                    valid_err2 = pred_error(f_pred, valid_set_x2, valid_set_y2, kf_valid2)
                    test_err2 = pred_error(f_pred, test_set_x2, test_set_y2, kf_test2)
                    train_err3 = pred_error(f_pred, train_set_x3, train_set_y3, kf3)
                    valid_err3 = pred_error(f_pred, valid_set_x3, valid_set_y3, kf_valid3)
                    test_err3 = pred_error(f_pred, test_set_x3, test_set_y3, kf_test3)
                    train_err4 = pred_error(f_pred, train_set_x4, train_set_y4, kf4)
                    valid_err4 = pred_error(f_pred, valid_set_x4, valid_set_y4, kf_valid4)
                    test_err4 = pred_error(f_pred, test_set_x4, test_set_y4, kf_test4)
                    train_err = pred_error(f_pred, train_set_x, train_set_y, kf, add_num=True)
                    valid_err = pred_error(f_pred, valid_set_x, valid_set_y, kf_valid, add_num=True)
                    test_err = pred_error(f_pred, test_set_x, test_set_y, kf_test, add_num=True)
                    
                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= numpy.array(history_errs)[:, 0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0
                        re_counter = 0

                    print( ('Train1 ', train_err1, 'Valid1 ', valid_err1,
                           'Test1 ', test_err1) )
                    print( ('Train2 ', train_err2, 'Valid2 ', valid_err2,
                           'Test2 ', test_err2) )
                    print( ('Train3 ', train_err3, 'Valid3 ', valid_err3,
                           'Test3 ', test_err3) )
                    print( ('Train4 ', train_err4, 'Valid4 ', valid_err4,
                           'Test4 ', test_err4) )
                    # print( ('Train_ave ', (train_err1*3561 + train_err2*2275 + train_err3*2269 + train_err4*439)/8544, 
                           # 'Valid_ave ', (valid_err1*453 + valid_err2*304 + valid_err3*293 + valid_err4*51)/1101,
                           # 'Test_ave ', (test_err1*898 + test_err2*608 + test_err3*589 + test_err4*115)/2210 ) )
                    print( ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err) )
                           
                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience, 0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break
                    print('++++++++++++++++++++++++++',bad_counter,'++++++++++++++++++++++++++++++++')
                    
                    if (len(history_errs) > pat and
                        valid_err >= numpy.array(history_errs)[:-pat, 0].min()):
                        re_counter += 1
                        #if re_counter > pat and learning_rate>1e-5:
                        if re_counter > pat:
                            print('Reduce Learning_rate!')
                            learning_rate = learning_rate / 10
                            print('learning_rate ', learning_rate)
                            break
                    
                    
            print('Seen %d samples' % n_samples)

            #if eidx>1 and eidx%1000==0 :
            #    learning_rate = learning_rate / 10

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted1 = get_minibatches_idx(0, train_num1, batch_size)
    kf_train_sorted2 = get_minibatches_idx(1, train_num2, batch_size)
    kf_train_sorted3 = get_minibatches_idx(2, train_num3, batch_size)
    kf_train_sorted4 = get_minibatches_idx(3, train_num4, batch_size)
    kf_train_sorted = kf_train_sorted1 + kf_train_sorted2 + kf_train_sorted3 + kf_train_sorted4
    train_err = pred_error(f_pred, train_set_x, train_set_y, kf_train_sorted, add_num=True)
    valid_err = pred_error(f_pred, valid_set_x, valid_set_y, kf_valid, add_num=True)
    test_err = pred_error(f_pred, test_set_x, test_set_y, kf_test, add_num=True)

    print( 'Train error ', train_err, 'Valid error ', valid_err, 'Test error ', test_err )
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err


if __name__ == '__main__':
    f = open('data/out/disease_cp_k3_new_out1.txt', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    evaluate_model()
    f.close()








