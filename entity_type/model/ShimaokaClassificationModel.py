# -*- coding: utf-8 -*-

import tensorflow as tf
from base_model import Model
import numpy as np
import layers as layers_lib

class ShimaokeModel(Model):
  def __init__(self,model_args):
    super(ShimaokeModel, self).__init__()
    self.args = model_args.args
    
    self.vocab_size = model_args.vocab_size
    self.batch_size=self.args.batch_size
    self.trieTree = model_args.type_tree
    self._init_placeholder()
    self._init_embed()
    self._init_main()
    self._init_optimizer()
      
  def _init_placeholder(self):
    self.word_embed_pl = tf.placeholder(tf.float32,[self.vocab_size,self.args.word_dim])
    
    self.input_data = tf.placeholder(tf.int32,[None,self.args.feature_length],name='inputdata')
    
    
    self.output_data = tf.placeholder(tf.int32,[None,self.args.class_size],name='inputdata')
    
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
    
    
  def _init_embed(self):
    self.word_embed_matrix = tf.Variable(initial_value=self.word_embed_pl,trainable=False)
    
    self.layers={}
    self.layers['BiLSTM'] = layers_lib.BiLSTM(self.args.rnn_size)
    
    self.layers['att_weights'] = {
    'h1': tf.Variable(tf.truncated_normal([2*self.args.rnn_size,self.args.attention_hidden_size],stddev=0.01)),
    'h2': tf.Variable(tf.truncated_normal([self.args.attention_hidden_size,1],stddev=0.01)),
    }
    
    self.input_data_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.input_data)
    

  def _init_main(self):
    self.prediction_1,self.loss = self.cl_loss_fn(self.input_data_embed)
    print('self.loss:',self.loss)
    
    self.prediction = tf.sigmoid(self.prediction_1)
    
    
  def _init_optimizer(self):
    self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.tvars if 'biase' not in v.name])
    self.total_loss = self.loss + self.l2_loss*self.args.l2_loss_w
    optimizer = tf.train.AdamOptimizer(self.args.learning_rate)          
    self.train_pos_op = optimizer.minimize(self.total_loss)
    
      
  def cl_loss_fn(self,embedded,return_intermediate=False):

    self.reshape_input = tf.concat([tf.reshape(embedded,[-1,self.args.word_dim]),tf.constant(np.zeros((1,self.args.word_dim),dtype=np.float32))],0)
    
    
    input_f1_embed = tf.nn.embedding_lookup(self.reshape_input,self.entMentIndex)
    input_f2_embed = tf.nn.embedding_lookup(self.reshape_input,self.entCtxLeftIndex)
    input_f3_embed = tf.nn.embedding_lookup(self.reshape_input,self.entCtxRightIndex)
    
    input_f1_embed = tf.nn.l2_normalize(input_f1_embed,-1)  
    input_f2_embed = tf.nn.l2_normalize(input_f2_embed,-1)
    input_f3_embed = tf.nn.l2_normalize(input_f3_embed,-1)
      
    input_f1 = tf.divide(tf.reduce_sum(input_f1_embed,1),
                        tf.cast(self.entMentLent,tf.float32))
    
    input_f2,_,_,_ =self.layers['BiLSTM'](input_f2_embed,tf.cast(self.ctxLeftLent[:,0],tf.int32),self.keep_prob)
    
    #input_f3_embed = tf.contrib.layers.batch_norm(input_f3_embed,is_training=self.is_training)
    input_f3,_,_,_ = self.layers['BiLSTM'](input_f3_embed,tf.cast(self.ctxRightLent[:,0],tf.int32),self.keep_prob)
    
    
    input_ctx_all = tf.concat([input_f2,input_f3],1)
    
    '''
    @2018/1/17 do not conclude the entity mention and weight balance
    '''
    att_w1 = tf.nn.tanh(tf.einsum('aij,jk->aik',input_ctx_all,self.layers['att_weights']['h1']))
    self.att_w2 = tf.nn.softmax(tf.einsum('aij,jk->aik',att_w1,self.layers['att_weights']['h2'])[:,:,0],-1)
    
    att_w = tf.tile(tf.expand_dims(self.att_w2,-1),[1,1,2*self.args.rnn_size])
    
    ctx_features = tf.reduce_sum(tf.multiply(input_ctx_all , att_w),1)
    
    if self.args.dropout:
      ctx_features =  tf.nn.dropout(ctx_features,self.keep_prob)
      input_f1 =  tf.nn.dropout(input_f1,self.keep_prob)
    
    '''
    @2018/1/17 logistic regression, different from our 2 layer MLPs.
    '''
    self.input_total = tf.concat([input_f1,ctx_features],1)
    
    prediction = tf.contrib.layers.fully_connected(self.input_total,self.args.class_size,activation_fn=None)
    
    loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,
                                                                      labels=tf.cast(self.output_data,tf.float32)),
                              reduction_indices=1))
    return prediction,loss


      