# -*- coding: utf-8 -*-


import tensorflow as tf
from base_model import Model
import numpy as np
import layers as layers_lib

class ComplExRandomModel(Model):
  def __init__(self,model_args):
    super(ComplExRandomModel, self).__init__()
    self.args = model_args.args
    self.type_tree= model_args.type_tree
    self.vocab_size = model_args.vocab_size
    self.batch_size=self.args.batch_size
    self.training_data_nums = model_args.training_data_nums
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
    
    #@time: 2019/1/21
    self.complEx_type = tf.placeholder(tf.int32,[None,1],name='complEx_type')
    self.complEx_pos_neg_type = tf.placeholder(tf.int32,[None,21],name='complEx_pos_neg_type')
    self.complEx_pos_neg_type_tag = tf.placeholder(tf.int32,[None,21],name='complEx_pos_neg_type_tag')
  
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
  
  def _gen_struct_loss(self):
    self.complEx_type_rel = tf.nn.embedding_lookup(self.rel_type_embedding,self.complEx_type)
    self.complEx_type_img = tf.nn.embedding_lookup(self.img_type_embedding,self.complEx_type)
    
    self.complEx_pos_neg_type_rel = tf.nn.embedding_lookup(self.rel_type_embedding,
                                                           self.complEx_pos_neg_type)
    self.complEx_pos_neg_type_img = tf.nn.embedding_lookup(self.img_type_embedding,
                                                           self.complEx_pos_neg_type)

    
    self.pos_neg_pair_score = tf.einsum('aij,ajk->aik', tf.einsum('aij,jk->aik',self.complEx_type_rel,self.rel_isa_embedding),
                                    tf.transpose(self.complEx_pos_neg_type_rel,[0,2,1])
                                    ) + \
                          tf.einsum('aij,ajk->aik', tf.einsum('aij,jk->aik',self.complEx_type_rel,self.img_isa_embedding),
                                    tf.transpose(self.complEx_pos_neg_type_img,[0,2,1])
                                    ) + \
                          tf.einsum('aij,ajk->aik', tf.einsum('aij,jk->aik',self.complEx_type_img,self.rel_isa_embedding),
                                    tf.transpose(self.complEx_pos_neg_type_img,[0,2,1])
                                    ) - \
                          tf.einsum('aij,ajk->aik', tf.einsum('aij,jk->aik',self.complEx_type_img,self.img_isa_embedding),
                                    tf.transpose(self.complEx_pos_neg_type_rel,[0,2,1])
                                    )
    self.pos_neg_pair_score=self.pos_neg_pair_score[:,0,:]
    print('pos_pair_score:',self.pos_neg_pair_score)
    
    self.loss_struct = tf.reduce_mean(tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pos_neg_pair_score,
                                                labels=tf.cast(self.complEx_pos_neg_type_tag,tf.float32))
        ,-1),-1)
                       
    print('loss_struct:',self.loss_struct)
    
    print('----------------------------')
    
  def _init_embed(self):
    self.word_embed_matrix = tf.Variable(initial_value=self.word_embed_pl,trainable=False)
    
    self.layers={}
    self.layers['BiLSTM'] = layers_lib.BiLSTM(self.args.rnn_size)
    self.feature_dims = self.args.attention_hidden_size
    self.layers['att_weights'] = {
    'h1': tf.Variable(tf.truncated_normal([2*self.args.rnn_size,self.args.attention_hidden_size],stddev=0.01)),
    'h2': tf.Variable(tf.truncated_normal([self.args.attention_hidden_size,1],stddev=0.01)),
    }
    
    self.input_data_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.input_data)
    #time:2019/1/21
    self.rel_type_embedding = tf.Variable(   
          initial_value = tf.truncated_normal([self.args.class_size,self.args.type_dim],stddev=0.01),
          trainable=True,
          name="rel_type_embedding",
          dtype = tf.float32   
    )
    
    self.img_type_embedding = tf.Variable(   
          initial_value = tf.truncated_normal([self.args.class_size,self.args.type_dim],stddev=0.01),
          trainable=True,
          name="img_type_embedding",
          dtype = tf.float32   
    )
    
    self.rel_isa_embedding_diag = tf.Variable(   
          initial_value = tf.truncated_normal([self.args.type_dim],stddev=0.01),
          trainable=True,
          name="img_type_embedding",
          dtype = tf.float32   
    )
    
    self.rel_isa_embedding = tf.diag(self.rel_isa_embedding_diag)
    
    
    self.img_isa_embedding_diag = tf.Variable(   
          initial_value = tf.truncated_normal([self.args.type_dim],stddev=0.01),
          trainable=True,
          name="img_isa_embedding_diag",
          dtype = tf.float32   
    )
    
    self.img_isa_embedding = tf.diag(self.img_isa_embedding_diag)
    
    if self.args.is_training == True:
      self.train_type_weight_l1 = tf.Variable(   
            initial_value = tf.truncated_normal([self.training_data_nums,self.args.max_pos_type_l1],stddev=0.01),
            trainable=True,
            name="train_type_weight_l1",
            dtype = tf.float32   
      )
      
      self.train_type_weight_l2 = tf.Variable(   
              initial_value = tf.truncated_normal([self.training_data_nums,self.args.max_pos_type_l1,self.args.max_pos_type_l2],stddev=0.01),
              trainable=True,
              name="train_type_weight_l2",
              dtype = tf.float32   
        )
      
      if self.type_tree.hieral_layer==3:
        self.train_type_weight_l3 = tf.Variable(   
            initial_value = tf.truncated_normal([self.training_data_nums,self.args.max_pos_type_l1,self.args.max_pos_type_l2,
                                                 self.args.max_pos_type_l3],stddev=0.01),
            trainable=True,
            name="train_type_weight_l3",
            dtype = tf.float32   
        )
    
  def _init_main(self):
    self.prediction_1,self.loss = self.cl_loss_fn(self.input_data_embed)
    print('self.loss:',self.loss)
    
    self.prediction = tf.sigmoid(self.prediction_1)
    
  def _init_optimizer(self):
    self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print('tvars:',self.tvars)
    self._gen_struct_loss()
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.tvars if 'biase' not in v.name])
    #whether we need to refinforcement the structure loss functions~~~~
    struct_w=1.0
    self.total_loss = self.loss + self.l2_loss*self.args.l2_loss_w + struct_w*self.loss_struct
    
    optimizer = tf.train.AdamOptimizer(self.args.learning_rate)          
    self.train_pos_op = optimizer.minimize(self.total_loss)
    
      
  def cl_loss_fn(self,embedded,return_intermediate=False):

    self.reshape_input = tf.concat([tf.reshape(embedded,[-1,self.args.word_dim]),tf.constant(np.zeros((1,self.args.word_dim),dtype=np.float32))],0)
    
    
    input_f1_embed = tf.nn.embedding_lookup(self.reshape_input,self.entMentIndex)
    input_f2_embed = tf.nn.embedding_lookup(self.reshape_input,self.entCtxLeftIndex)
    input_f3_embed = tf.nn.embedding_lookup(self.reshape_input,self.entCtxRightIndex)
    
    input_f1 = tf.divide(tf.reduce_sum(input_f1_embed,1),
                        tf.cast(self.entMentLent,tf.float32))
    
    input_f2,_,_,_ =self.layers['BiLSTM'](input_f2_embed,tf.cast(self.ctxLeftLent[:,0],tf.int32),self.keep_prob)
    
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
    
    #we need to embedding the input_total into the img and rel
    self.ment_feature_rel = tf.layers.dense(self.input_total,self.args.type_dim,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    self.ment_feature_img = tf.layers.dense(self.input_total,self.args.type_dim,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    print('ment_feature_img:',self.ment_feature_img)
    
    prediction = tf.matmul(self.ment_feature_rel,self.rel_type_embedding,transpose_b=True) +\
                 tf.matmul(self.ment_feature_img,self.img_type_embedding,transpose_b=True)
    '''
    loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,
                                        labels=tf.cast(self.output_data,tf.float32)),
                              reduction_indices=1))'''
    
    if self.args.is_training == True:
      self.pos_types_rel_embed = tf.nn.embedding_lookup(self.rel_type_embedding,self.pos_types) #(batch_size,max_pos_type,args.type_dim)
      self.pos_types_img_embed = tf.nn.embedding_lookup(self.img_type_embedding,self.pos_types)
      
      self.neg_types_rel_embed = tf.nn.embedding_lookup(self.rel_type_embedding,self.neg_types)
      self.neg_types_img_embed = tf.nn.embedding_lookup(self.img_type_embedding,self.neg_types)
      
      if self.type_tree.hieral_layer==3:
        self.a_norm = self._get_type_weight_2()
      else:
        self.a_norm = self._get_type_weight()
      
      if self.type_tree.hieral_layer==2:
        self.a_norm = tf.multiply(self.a_norm,tf.reshape(self.pos_mask_l2,[-1,self.args.max_pos_type]))
      else:
        self.a_norm = tf.multiply(self.a_norm,tf.reshape(self.pos_mask_l3,[-1,self.args.max_pos_type]))
        
      self.a_norm = tf.div(self.a_norm,tf.tile(tf.reduce_sum(self.a_norm,-1,keepdims=True),[1,self.args.max_pos_type]))
      
      self.pos_types_score = tf.einsum('aij,ajk->aik',self.pos_types_rel_embed,tf.expand_dims(self.ment_feature_rel,-1)) +\
                             tf.einsum('aij,ajk->aik',self.pos_types_img_embed,tf.expand_dims(self.ment_feature_img,-1))
                 
      print('pos_types_score:',self.pos_types_score)
      self.ent_w_pos_type = tf.reduce_sum(tf.multiply(self.a_norm,self.pos_types_score[:,:,0]),-1)
      print('ent_w_pos_type:',self.ent_w_pos_type)
      self.ent_w_neg_type = tf.einsum('aij,ajk->aik',self.neg_types_rel_embed,tf.expand_dims(self.ment_feature_rel,-1)) +\
                             tf.einsum('aij,ajk->aik',self.neg_types_img_embed,tf.expand_dims(self.ment_feature_img,-1))
      print('ent_w_neg_type:',self.ent_w_neg_type)
      loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(self.args.margin
                              -tf.tile(tf.expand_dims(self.ent_w_pos_type,-1),[1,self.args.max_neg_type])
                             +self.ent_w_neg_type[:,:,0]),-1))
    else:
      loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,
                                        labels=tf.cast(self.output_data,tf.float32)),
                              reduction_indices=1))
      
    return prediction,loss
  def _get_renorm(self,type_weight_embed,type_weight_mask,dims,axis_num=1):
    type_weight_softmax = tf.nn.softmax(type_weight_embed,-1)
    
    type_weight = tf.multiply(type_weight_softmax,type_weight_mask)
    
    if '1.4.0' in tf.__version__ :
      type_weight_sum = tf.reciprocal(tf.reduce_sum(type_weight,-1,keep_dims=True)+1e-6)
    else:
      type_weight_sum = tf.reciprocal(tf.reduce_sum(type_weight,-1,keepdims=True)+1e-6)
      
    if axis_num==1:
      type_weight_fianl = tf.multiply(type_weight,tf.tile(type_weight_sum,[1,dims]))
    elif axis_num==2:
      type_weight_fianl = tf.multiply(type_weight,tf.tile(type_weight_sum,[1,1,dims]))
    elif axis_num==3:
      type_weight_fianl = tf.multiply(type_weight,tf.tile(type_weight_sum,[1,1,1,dims]))
    else:
      print('wrong axis nums...')
    return type_weight_fianl
  
  def _get_type_weight(self):
    type_weight_random_embed_l1 = self._get_renorm(tf.nn.embedding_lookup(self.train_type_weight_l1,self.ent_rel_index),
                                                   self.pos_mask_l1,self.args.max_pos_type_l1,1)
    type_weight_random_embed_l2 = self._get_renorm(tf.nn.embedding_lookup(self.train_type_weight_l2,self.ent_rel_index),
                                                   self.pos_mask_l2,self.args.max_pos_type_l2,2)
    
    
    type_weight_random_final = tf.multiply(tf.tile(tf.expand_dims(type_weight_random_embed_l1,-1),[1,1,self.args.max_pos_type_l2]),type_weight_random_embed_l2)
    
    print('type_weight_random_final:',type_weight_random_final)
    
    type_weight= tf.reshape(type_weight_random_final,[-1,self.args.max_pos_type])
   
    return type_weight
 
  def _get_type_weight_2(self):
    
    type_weight_random_embed_l1 = self._get_renorm(tf.nn.embedding_lookup(self.train_type_weight_l1,self.ent_rel_index),
                                                   self.pos_mask_l1,self.args.max_pos_type_l1,1)
    
    type_weight_random_embed_l2 = self._get_renorm(tf.nn.embedding_lookup(self.train_type_weight_l2,self.ent_rel_index),
                                                   self.pos_mask_l2,self.args.max_pos_type_l2,2)
    
    type_weight_random_embed_l3 = self._get_renorm(tf.nn.embedding_lookup(self.train_type_weight_l3,self.ent_rel_index),
                                                   self.pos_mask_l3,self.args.max_pos_type_l3,3)
    
    
    type_weight_random_l2 = tf.multiply(tf.tile(tf.expand_dims(type_weight_random_embed_l1,-1),[1,1,self.args.max_pos_type_l2]),type_weight_random_embed_l2)

    type_weight_random_l3 =tf.multiply(tf.tile(tf.expand_dims(type_weight_random_l2,-1),[1,1,1,self.args.max_pos_type_l3]),type_weight_random_embed_l3)
    
    type_weight = tf.reshape(type_weight_random_l3,[-1,self.args.max_pos_type])
    
    return type_weight