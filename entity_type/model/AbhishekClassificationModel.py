# -*- coding: utf-8 -*-


import tensorflow as tf
from base_model import Model
import numpy as np
import layers as layers_lib

class AbhishekModel(Model):
  def __init__(self,model_args):
    super(AbhishekModel, self).__init__()
    self.model_args = model_args
    self.args = model_args.args
    self.vocab_size = model_args.vocab_size
    self.batch_size=self.args.batch_size
   
    self._init_placeholder()
    self._init_embed()
    self._init_main()
    self._init_optimizer()
      
  def _init_placeholder(self):
    self.word_embed_pl = tf.placeholder(tf.float32,[self.vocab_size,self.args.word_dim])
    
    self.input_data = tf.placeholder(tf.int32,[None,self.args.feature_length],name='inputdata')
    
    
    self.output_data = tf.placeholder(tf.int32,[None,self.args.class_size],name='inputdata')
    
    self.pos_types = tf.placeholder(tf.int32,[None,self.args.max_pos_type] ,name='pos_types') 
    self.pos_mask = tf.placeholder(tf.float32,[None,self.args.max_pos_type],name='pos_mask')
    
    #we need to record the weights..
    self.ent_rel_index = tf.placeholder(tf.int32,[None])
    
    self.neg_types = tf.placeholder(tf.int32,[None,self.args.max_neg_type] ,name='neg_type')  #sample one negative instance
    
    #we need to add type-type pair 
    self.type_pair = tf.placeholder(tf.int32,[None,2])
    self.type_cor_score = tf.placeholder(tf.float32,[None])
    
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
    self.char_embed = tf.Variable(
        initial_value = tf.truncated_normal([len(self.model_args.characterEmbed.character2id),self.args.char_rnn_size],stddev=0.01),
        trainable=True,
        name="char_embeddings",
        dtype = tf.float32   
        )
    
    self.word_embed_matrix = tf.Variable(initial_value=self.word_embed_pl,trainable=False)
    
    self.layers={}
    self.layers['BiLSTM'] = layers_lib.BiLSTM(self.args.rnn_size)
    self.layers['LSTM'] = layers_lib.LSTM(self.args.char_rnn_size)  #随便设置的char的
    
    self.label_embedding = tf.get_variable('label_embeddings',
                                          initializer=tf.truncated_normal(
                                              [500,self.args.class_size],
                                              stddev=0.01),
                                          trainable=True)

    self.input_data_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.input_data)
    

  def _init_main(self):
    ent_ment_chars_embed = tf.nn.embedding_lookup(self.char_embed,self.entment_chars)
    _,self.ent_ment_chars_feature =  self.layers['LSTM'](ent_ment_chars_embed,self.entment_chars_lent)
    print('self.ent_ment_chars_feature :',self.ent_ment_chars_feature )
    
    self.prediction,self.loss = self.cl_loss_fn(self.input_data_embed)
    print('self.loss:',self.loss)
    
    
  def _init_optimizer(self):
    self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.tvars if 'biase' not in v.name])
    self.total_loss = self.loss + self.l2_loss*self.args.l2_loss_w
    optimizer = tf.train.AdamOptimizer(self.args.learning_rate)          
    self.train_pos_op = optimizer.minimize(self.total_loss)
    
      
  def cl_loss_fn(self,embedded,return_intermediate=False):

    self.reshape_input = tf.concat([tf.reshape(embedded,[-1,self.args.word_dim]),tf.constant(np.zeros((1,self.args.word_dim),dtype=np.float32))],0)
  
    input_f2_embed = tf.nn.embedding_lookup(self.reshape_input,self.entCtxLeftIndex)
    input_f3_embed = tf.nn.embedding_lookup(self.reshape_input,self.entCtxRightIndex)
    
    input_f1= self.ent_ment_chars_feature[1]
    print('input_f1:',input_f1)
    
    _,_,input_f2,_, =self.layers['BiLSTM'](input_f2_embed,tf.cast(self.ctxLeftLent[:,0],tf.int32),self.keep_prob)
    print('input_f2:',input_f2)
    

    _,_,input_f3,_ = self.layers['BiLSTM'](input_f3_embed,tf.cast(self.ctxRightLent[:,0],tf.int32),self.keep_prob)
    print('input_f3:',input_f3)
    
    ctx_features = tf.concat([input_f2,input_f3],1)
    
    if self.args.dropout:
      ctx_features =  tf.nn.dropout(ctx_features,self.keep_prob)
      
    self.input_total = tf.concat([input_f1,ctx_features],1)
    
    prediction_l1 =tf.contrib.layers.fully_connected(self.input_total,500,activation_fn=None,biases_regularizer=None)
    print('prediction_l1:',prediction_l1)
    
    prediction = tf.einsum('ij,jk->ik',prediction_l1,self.label_embedding)
    
    #hinge_loss
    labels = tf.cast(self.output_data,tf.float32)
    logits =  prediction
    
    pos = tf.reduce_sum(tf.multiply(labels,tf.nn.relu(1-logits)),-1)
    neg = tf.reduce_sum(tf.multiply(1.0-labels,tf.nn.relu(1+logits)),-1)
    loss = tf.reduce_mean(pos + neg)  #用reduce_sum对论文
    
    return prediction,loss


      