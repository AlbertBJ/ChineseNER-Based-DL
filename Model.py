__author__ = 'Albert Wang'
import tensorflow as tf
import numpy as np
# import data_util as data
import math


class Model:
    def __init__(self, dic_config):
        self.lr = dic_config['lr']
        self.epoch = dic_config['epoch']
        self.voc_size = dic_config['voc_size']
        self.embedding_dim = dic_config['embedding_dim']
        self.pre_embedding = dic_config['embedding']
        self.usePreTrained = dic_config['usePreTrained']
        self.senMaxLen = dic_config['sentenceMaxLen']
        self.tag_size = dic_config['tag_size']
        self.batch_size = dic_config['batch_size']
        self.is_train = dic_config['is_train']
        if self.is_train:
            self.rate = dic_config['rate']
        else:
            self.rate = 0

        # default list
        self.id2word = {}
        self.id2tag = {}
        self.word2id = {}
        self.tag2id = {}

    def buildNN(self):
        self.inputdata = tf.placeholder(
            tf.int32, shape=[None, self.senMaxLen], name='input_data')
        self.tagdata = tf.placeholder(
            tf.int32, shape=[None, self.senMaxLen], name='label_data')
        self.sequence_length = tf.placeholder(
            tf.int32, shape=[None], name='real_length')
        l2_regularizer = None
        if self.is_train:
            l2_regularizer = tf.contrib.layers.l2_regularizer(0.0001)

        # get embedding matrix for next step(BiLSTM layer)
        lstm_input = self.embeddingLayer()

        # get bilstm-output for next step(dense layer)
        lstm_output = self.biLSTMLayer(lstm_input)

        # get dense_output for next step(crf layer)
        dense_output = self.denseLayer(lstm_output, l2_regularizer)

        # get log_likelihood for next step(loss layer)
        log_likelihood = self.CRFLayer(dense_output)

        # get loss
        self.lossLayer(log_likelihood)

    def embeddingLayer(self):
        '''
        embedding layer
        '''

        # get embedding voctor if embedding vector existed
        with tf.variable_scope("embedding"):
            word_em = tf.get_variable('word_embedding',
                                      [self.voc_size, self.embedding_dim])
            if self.usePreTrained and self.is_train:  # use pre-trained embedding vector
                word_em.assign(self.pre_embedding)
            lstm_input = tf.nn.embedding_lookup(word_em, self.inputdata)
            lstm_input = tf.nn.dropout(lstm_input, rate=self.rate)

            return lstm_input

    def biLSTMLayer(self, inputs):
        '''
        BI-LSTM layer
        '''
        with tf.variable_scope('lstm'):
            fw = tf.contrib.rnn.LSTMBlockCell(self.embedding_dim, name='fw')
            bw = tf.contrib.rnn.LSTMBlockCell(self.embedding_dim, name='bw')
            # output shape is [batch_size,max_time,cell_output_size]
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                fw, bw, inputs, dtype=tf.float32)
            lstm_output = tf.concat([output_fw, output_bw], axis=2)
            # new shape [x,self.embedding_dim*2] for calculating next z=xW+b
            lstm_output = tf.reshape(
                lstm_output, shape=[-1, self.embedding_dim * 2])
            return lstm_output

    def denseLayer(self, inputs, l2_regularizer):
        '''
        full connect layer
        '''
        with tf.variable_scope('full_connect'):
            w = tf.get_variable(
                'weight', [2 * self.embedding_dim, self.tag_size],
                dtype=tf.float32)
            b = tf.get_variable('bias', [self.tag_size], dtype=tf.float32)
            if not l2_regularizer or l2_regularizer is not None:
                tf.add_to_collection('l2_loss', l2_regularizer(w))

            dense_output = tf.nn.relu(tf.add(tf.linalg.matmul(inputs, w), b))
            # recover original shape for next crf
            dense_output = tf.reshape(dense_output,
                                      [-1, self.senMaxLen, self.tag_size])
            return dense_output

    def CRFLayer(self, inputs):
        '''
        get crf layer 
        '''
        with tf.variable_scope('crf'):
            log, transition = tf.contrib.crf.crf_log_likelihood(
                inputs,
                tag_indices=self.tagdata,
                sequence_lengths=self.sequence_length)
            self.decode_tags, self.best_score = tf.contrib.crf.crf_decode(
                inputs, transition, self.sequence_length)
        return log

    def lossLayer(self, log_likelihood):
        '''
        get loss layer minimizing loss
        '''
        # crf  maxmizing log_likelihood, so add a minus before log_likelihood here to minimize the loss
        self.loss = tf.math.reduce_mean(-log_likelihood) + tf.add_n(
            tf.get_collection('l2_loss'))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self):
        '''
        train model
        '''
        # process data

        # build NN

        # train

        pass

    def Inference(self):
        '''
        run model to inference results
        '''
        pass

    def precision(self) -> np.float32:
        pass

    def recall(self) -> np.float32:
        pass

    def F1(self):
        pass