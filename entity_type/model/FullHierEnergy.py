# -*- coding: utf-8 -*-

import tensorflow as tf
from base_model import Model
import numpy as np
import layers as layers_lib


class FullHierEnergy(Model):
  
  def __init__(self,model_args):
    super(FullHierEnergy, self).__init__()
    self.args = model_args.args
    self.type_tree= model_args.type_tree
    self.vocab_size = model_args.vocab_size
    self.training_data_nums = model_args.training_data_nums
    self.type_path = model_args.type_tree.type_path
    self.batch_size=self.args.batch_size
    self.stop_token_ids = self.type_tree.stop_token_ids
    self._init_placeholder()
    self._init_embed()
    self._init_main()

  def _init_embed(self):
    self.word_embed_matrix = tf.Variable(initial_value=self.word_embed_pl,trainable=False)
    
    self.layers={}
    self.layers['BiLSTM'] = layers_lib.BiLSTM(self.args.rnn_size)
    
    #This place is very important!
    self.feature_dims = self.args.attention_hidden_size
    
  
    self.layers['att_weights'] = {
    'h_m':tf.Variable(tf.truncated_normal([self.args.word_dim,self.feature_dims],stddev=0.01)),
    'h1': tf.Variable(tf.truncated_normal([2*self.args.rnn_size,self.feature_dims],stddev=0.01)),
    'h2': tf.Variable(tf.truncated_normal([self.feature_dims,1],stddev=0.01)),
    }
    
    
    self.type_token_embeddings = tf.Variable(   
          initial_value = tf.truncated_normal([self.type_tree.typeToken_num,self.args.type_dim],stddev=0.01),
          trainable=True,
          name="type_token_embeddings",
          dtype = tf.float32   
    )
    self.type_token_embeddings_add_padding = tf.concat([self.type_token_embeddings,tf.constant([[0.0]*self.args.type_dim])],0)
    
    self.type_path_embedding = tf.nn.embedding_lookup(self.type_token_embeddings_add_padding,self.type_path)
    self.type_path_embedding = tf.reduce_sum(self.type_path_embedding,1)
    
    self.type_embeddings_norm  = tf.nn.l2_normalize(self.type_path_embedding,-1)
    
    self.type_embeddings_norm_add_padding = tf.concat([self.type_embeddings_norm,tf.constant([[0.0]*self.args.type_dim])],0)
    
  def _init_main(self):
    self.input_data_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.input_data)
    self.ent_ctx_feature = self.get_ent_feature(self.input_data_embed)  #(batch_size,2*args.rnn_size+args.word_dim)
    
    self.ent_ctx_t1 = tf.nn.l2_normalize(tf.layers.dense(self.ent_ctx_feature,self.args.type_dim,tf.nn.relu),-1) 
    self.ent_w = tf.expand_dims(self.ent_ctx_t1,1)  #(batch_size,1,type_dims)  
    
    self.ent_w_all_type = tf.einsum('aij,jk->aik',self.ent_w,  #(batch_size,class_size)
                                    tf.transpose(self.type_embeddings_norm,[1,0]))[:,0,:]
    self.prediction = self.ent_w_all_type
    
    self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(self.tvars)
    
    if self.args.is_training == True:
      self.pos_types_embed = tf.nn.embedding_lookup(self.type_embeddings_norm_add_padding,self.pos_types) #(batch_size,max_pos_type,args.type_dim)
      self.neg_types_embed = tf.nn.embedding_lookup(self.type_embeddings_norm_add_padding,self.neg_types)
      
      #get weight a_norm
      if self.type_tree.hieral_layer==3:
        pos_mask = tf.reshape(self.pos_mask_l3,[-1,self.args.max_pos_type])
      else:
        pos_mask = tf.reshape(self.pos_mask_l2,[-1,self.args.max_pos_type])
      self.a_norm = tf.div(pos_mask,tf.reduce_sum(pos_mask,-1,keepdims=True))
      
      self.pos_types_a = tf.einsum('aij,ajk->aik',tf.transpose(self.pos_types_embed,[0,2,1]),
                                               tf.expand_dims(self.a_norm,-1))[:,:,0]
      
      self.ent_w_pos_type = tf.einsum('aij,ajk->aik',self.ent_w
                                    ,tf.expand_dims(self.pos_types_a,-1))[:,0,0] #(batch_size,)
    
      self.ent_w_neg_type = tf.einsum('aij,ajk->aik',self.ent_w,
                                    tf.transpose(self.neg_types_embed,[0,2,1]))[:,0,:]
    
      self.loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(self.args.margin
                              -tf.tile(tf.expand_dims(self.ent_w_pos_type,-1),[1,self.args.max_neg_type])
                             +self.ent_w_neg_type),-1))
      self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.tvars if 'bias' not in v.name])
      self.total_loss = self.loss + self.l2_loss*self.args.l2_loss_w
      
      tf.summary.scalar('stop_token_l2_loss',self.l2_loss)
      tf.summary.scalar('energy_loss',self.loss)
      tf.summary.scalar('total_loss',self.total_loss)
      
      optimizer = tf.train.AdamOptimizer(self.learning_rate)          
      self.train_pos_op = optimizer.minimize(self.total_loss,var_list=self.tvars)

  def get_ent_feature(self,embedded):
    self.reshape_input = tf.concat([tf.reshape(embedded,[-1,self.args.word_dim]),tf.constant(np.zeros((1,self.args.word_dim),dtype=np.float32))],0)
    
    ent_embed = tf.nn.embedding_lookup(self.reshape_input,self.entMentIndex)
    left_embed = tf.nn.embedding_lookup(self.reshape_input,self.entCtxLeftIndex)
    right_embed = tf.nn.embedding_lookup(self.reshape_input,self.entCtxRightIndex)
    
    #ent_embed = tf.nn.l2_normalize(ent_embed,-1)
    #left_embed = tf.nn.l2_normalize(left_embed,-1)
    #right_embed = tf.nn.l2_normalize(right_embed,-1)
    
    input_f1 = tf.divide(tf.reduce_sum(ent_embed,1),
                        tf.cast(self.entMentLent,tf.float32))
    
    input_f2,_,_,_ =self.layers['BiLSTM'](left_embed,tf.cast(self.ctxLeftLent[:,0],tf.int32),self.keep_prob)
    input_f3,_,_,_ = self.layers['BiLSTM'](right_embed,tf.cast(self.ctxRightLent[:,0],tf.int32),self.keep_prob)
    
    
    
    input_ctx_all = tf.concat([input_f2,input_f3],1)
    att_w_m = tf.einsum('aij,jk->aik',tf.expand_dims(input_f1,1),self.layers['att_weights']['h_m'])
    if self.args.datasets=='BBN':
      att_w1 = tf.nn.tanh(tf.einsum('aij,jk->aik',input_ctx_all,self.layers['att_weights']['h1'])+att_w_m)
    else:
      att_w1 = tf.nn.relu(tf.einsum('aij,jk->aik',input_ctx_all,self.layers['att_weights']['h1'])+att_w_m)
    
    self.att_w2 = tf.nn.softmax(tf.einsum('aij,jk->aik',att_w1,self.layers['att_weights']['h2'])[:,:,0],-1)
    
    att_w = tf.tile(tf.expand_dims(self.att_w2,-1),[1,1,2*self.args.rnn_size])
    
    ctx_features = tf.reduce_sum(tf.multiply(input_ctx_all , att_w),1)
    
    if self.args.dropout:
      input_f1 = tf.nn.dropout(input_f1,self.keep_prob)  #I do not, whether this is benefit for the training...
      ctx_features =  tf.nn.dropout(ctx_features,self.keep_prob)
    ent_ctx_concat = tf.concat([input_f1,ctx_features],1)

    return ent_ctx_concat
  
  def _init_placeholder(self):
    self.word_embed_pl = tf.placeholder(tf.float32,[self.vocab_size,self.args.word_dim])
    self.input_data = tf.placeholder(tf.int32,[None,self.args.feature_length],name='inputdata')
    self.output_data = tf.placeholder(tf.int32,[None,self.args.class_size],name='output_data')
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    
    if self.args.is_training == True:
      self.pos_types = tf.placeholder(tf.int32,[None,self.args.max_pos_type] ,name='pos_types')  
      self.ent_rel_index = tf.placeholder(tf.int32,[None])
      self.neg_types = tf.placeholder(tf.int32,[None,self.args.max_neg_type] ,name='neg_type')  #sample negative types
      
      if self.type_tree.hieral_layer==2:
        self.pos_mask_l1 = tf.placeholder(tf.float32,[None,self.args.max_pos_type_l1],name='pos_mask_l1') 
        self.pos_mask_l2 = tf.placeholder(tf.float32,[None,self.args.max_pos_type_l1,self.args.max_pos_type_l2],name='pos_mask_l2') #for args.max_pos_type = layer_1_max_type *layer_2_max_type
      else:
        self.pos_mask_l1 = tf.placeholder(tf.float32,[None,self.args.max_pos_type_l1],name='pos_mask_l1') 
        self.pos_mask_l2 = tf.placeholder(tf.float32,[None,self.args.max_pos_type_l1,self.args.max_pos_type_l2],name='pos_mask_l2') 
        self.pos_mask_l3 = tf.placeholder(tf.float32,[None,self.args.max_pos_type_l1,self.args.max_pos_type_l2,self.args.max_pos_type_l3],
                                          name='pos_mask_l3') # args.max_pos_type = layer_1_max_type *layer_2_max_type * layer_3_max_type

    self.entMentIndex = tf.placeholder(tf.int32,
                                       [None,5],
                                       name='ent_mention_index')
    
    self.learning_rate = tf.placeholder(tf.float32,
                                       name='learning_rate')
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