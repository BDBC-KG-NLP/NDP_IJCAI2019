# -*- coding: utf-8 -*-

import tensorflow as tf
from base_model import Model
import numpy as np
import layers as layers_lib

class PengModel(Model):
  def __init__(self,model_args):
    super(PengModel, self).__init__()
    self.args = model_args.args
    self.trieTree = model_args.type_tree
    self.vocab_size = model_args.vocab_size
    self.batch_size=self.args.batch_size

    self._init_placeholder()
    self._init_embed()
    self._init_main()
    self._init_optimizer()
      
  def _init_placeholder(self):
    self.word_embed_pl = tf.placeholder(tf.float32,[self.vocab_size,self.args.word_dim])
    
    self.input_data = tf.placeholder(tf.int32,[None,self.args.feature_length],name='inputdata')

    self.output_data = tf.placeholder(tf.int32,[None,self.args.class_size],name='output_data')
    
    self.input_sents = tf.placeholder(tf.int32,[None,self.args.sentence_length],name='input_sents')
    self.input_positions = tf.placeholder(tf.int32, [None, self.args.sentence_length], name="input_positions")
    #we need to record the weights..
    self.ent_rel_index = tf.placeholder(tf.int32,[None])
    
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    self.is_training = tf.placeholder(tf.bool,name='is_training')
    
    self.entMentIndex = tf.placeholder(tf.int32,
                                       [None,5],
                                       name='ent_mention_index')
    
    self.entCtxLeftIndex = tf.placeholder(tf.int32,
                                          [None,self.args.ctx_length],
                                          name='ent_ctxleft_index')
    
    self.entCtxRightIndex = tf.placeholder(tf.int32,
                                           [None,self.args.ctx_length],
                                           name='ent_ctxright_index')
    
    
    self.entMentLent = tf.placeholder(tf.float32,[None,1],name='entMentLent')
    self.entment_chars = tf.placeholder(tf.int32,[None,self.args.characterLent],name='entment_chars')
    self.entment_chars_lent = tf.placeholder(tf.int32,[None],name='entment_chars_lent')

    self.ctxLeftLent = tf.placeholder(tf.float32,[None,1],name='ctxLeftLent')
    self.ctxRightLent = tf.placeholder(tf.float32,[None,1],name='ctxRightLent')
    
    
    self.tune  = tf.Variable(self.trieTree.create_prior(self.args.alpha),trainable=False,dtype=tf.float32,name='tune')
    
  def _init_embed(self):
    self.word_embed_matrix = tf.Variable(initial_value=self.word_embed_pl,trainable=False)
    
    self.pos_embed_matrix = tf.Variable(initial_value=np.random.normal(0,0.1,(2*self.args.sentence_length+1,self.args.pos_dims)),trainable=True,dtype=tf.float32)
    
    
    self.layers={}
    self.layers['BiLSTM'] = layers_lib.BiLSTM(self.args.rnn_size)
    
    self.layers['att_weights'] = {
    'h1': tf.Variable(tf.truncated_normal([self.args.rnn_size,1],stddev=0.01)),
    }
    
    self.input_data_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.input_data)
    
    self.input_sents_embed_un = tf.nn.embedding_lookup(self.word_embed_matrix,self.input_sents)
    self.input_pos_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.input_sents)
    self.input_sents_embed = tf.concat([self.input_sents_embed_un, self.input_pos_embed], 2)
  
  def _init_main(self):
    self.prediction,self.loss = self.cl_loss_fn(self.input_data_embed)
    print('self.loss:',self.loss)
    
  def _init_optimizer(self):
    self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.tvars if 'biase' not in v.name])
    self.total_loss = self.loss + self.l2_loss*self.args.l2_loss_w
    optimizer = tf.train.AdamOptimizer(self.args.learning_rate)          
    self.train_pos_op = optimizer.minimize(self.total_loss)
    
  def extract_last_relevant(self, outputs, seq_len):
    batch_size = tf.shape(outputs)[0]
    max_length = int(outputs.get_shape()[1])
    num_units = int(outputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (seq_len - 1)
    flat = tf.reshape(outputs, [-1, num_units])
    relevant = tf.gather(flat, index)
    return relevant
  
  def cl_loss_fn(self,embedded,return_intermediate=False):
    
    self.reshape_input = tf.concat([tf.reshape(embedded,[-1,self.args.word_dim]),tf.constant(np.zeros((1,self.args.word_dim),dtype=np.float32))],0)
    
    input_f1_embed = tf.nn.embedding_lookup(self.reshape_input,self.entMentIndex)
    self.input_f1 = tf.divide(tf.reduce_sum(input_f1_embed,1),
                        tf.cast(self.entMentLent,tf.float32))
    
    input_f1_embed = tf.nn.l2_normalize(input_f1_embed,-1)
    cell = tf.contrib.rnn.LSTMCell(self.args.rnn_size)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob,  output_keep_prob=self.keep_prob)
    
    
    outputs, states = tf.nn.dynamic_rnn(
                cell, input_f1_embed,
                sequence_length=self.entMentLent[:,0], dtype=tf.float32)
    self.men_repr = self.extract_last_relevant(outputs, tf.cast(self.entMentLent,tf.int32))[:,0,:]
    print('men_repr:',self.men_repr)
    
    with tf.device('/gpu:1'):
      _,input_sent_lstm,_,_ =self.layers['BiLSTM'](self.input_sents_embed,keep_prob=self.keep_prob)
      print('input_sent_lstm:',input_sent_lstm)
    
    att_word = tf.nn.softmax(tf.einsum('aij,jk->aik',tf.nn.tanh(input_sent_lstm),self.layers['att_weights']['h1']))
    print('att_word:',att_word)
    
    ctx_features=tf.einsum('aij,ajk->aik',tf.transpose(att_word,[0,2,1]),input_sent_lstm)[:,0,:]
    print('ctx_features:',ctx_features)
    
    self.input_total = tf.concat([self.input_f1,self.men_repr,ctx_features],-1)
    print('input_total:',self.input_total)
    
#    self.h_drop = tf.nn.dropout(tf.nn.relu(self.input_total), self.keep_prob)
#    self.h_drop = tf.layers.batch_normalization(self.h_drop, training=self.is_training)
    
    proba = tf.nn.softmax(tf.contrib.layers.fully_connected(self.input_total,self.args.class_size,activation_fn=None))
    adjusted_prob = tf.matmul(proba,self.tune,transpose_b= True)
    adjusted_proba = tf.clip_by_value(adjusted_prob, 1e-10, 1.0)

    target = tf.argmax(tf.multiply(adjusted_prob, tf.cast(self.output_data,tf.float32)), axis=1)
    target_index = tf.one_hot(target, self.args.class_size)
    
    loss = tf.reduce_mean(-tf.reduce_sum(target_index * tf.log(adjusted_proba), 1))
    
    predictions = tf.argmax(adjusted_proba, 1, name="predictions")
    print('predictions:',predictions)
    
    '''
    loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,
                                                                      labels=tf.cast(self.output_data,tf.float32)),
                              reduction_indices=1))'''
    return adjusted_proba,loss