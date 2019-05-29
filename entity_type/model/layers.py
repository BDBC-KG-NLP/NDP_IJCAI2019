# -*- coding: utf-8 -*-
"""
@function: generate MLP,LSTM layers
"""
import tensorflow as tf


def relu(x, alpha=0., max_value=None):
  _FLOATX = tf.float32
  '''ReLU.

  alpha: slope of negative section.
  '''
  negative_part = tf.nn.relu(-x)
  x = tf.nn.relu(x)
  if max_value is not None:
      x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                           tf.cast(max_value, dtype=_FLOATX))
  x -= tf.constant(alpha, dtype=_FLOATX) * negative_part
  return x

   
  
class CNN(object):
  '''
  CNN layer for Text
  '''
  def __init__(self,filters,word_embedding_size,num_filters):
    self.filters = filters
    self.embedding_size  = word_embedding_size
    self.num_filters = num_filters
    self.Ws = []
    self.bs = []
    for i,filter_size in enumerate(self.filters):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        filter_shape =[filter_size,self.embedding_size,1,self.num_filters]
        self.Ws.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="rnn_weight"+str(i)))
        self.bs.append(tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="rnn_bias"+str(i)))
  '''
  x: [batch,sequence_length,feature_dimension,1] === >[batch,in_height,in_width,in_channels]
  filter: [filter_height,filter_widht,in_channels,out_channels]
  '''
  def __call__(self,x,sequence_length):
    self.pooled_outputs = []
    for i,filter_size in enumerate(self.filters):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        #convolution layer
        W = self.Ws[i]; b = self.bs[i]
        conv = tf.nn.conv2d(
            x,
            W,
            strides=[1,1,1,1],
            padding='VALID',
            name='conv'+str(i))
        print conv
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        print h
         # Max-pooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize = [1, sequence_length-filter_size + 1, 1, 1],
            strides=[1,2,2,1],
            padding='VALID',
            name='pool')
        
    
        self.pooled_outputs.append(pooled)
    print 'self.pooled_outputs:',self.pooled_outputs
    #Combine all the pooled features
    num_filters_total = self.num_filters * len(self.filters) #?a little weird
    self.h_pool = tf.concat(self.pooled_outputs,3)
    self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
    
    return self.h_pool_flat
        
class LSTM(object):
  def __init__(self,cell_size,num_layers=1,name='LSTM'):
    self.cell_size = cell_size
    self.num_layers = num_layers
    self.reuse = None
    self.trainable_weights = None
    self.name = name
    
  def __call__(self,x,seq_length=None,keep_prob=1.0):
    with tf.variable_scope(self.name,reuse = self.reuse) as vs:
      self.cell =tf.contrib.rnn.LSTMCell(self.cell_size,reuse=tf.get_variable_scope().reuse)
      
      
      if seq_length ==None:  #get the real sequence length (suppose that the padding are zeros)
        used = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2))
        seq_length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)
      
      self.output,last_state=tf.contrib.rnn.static_rnn(self.cell,tf.unstack(tf.transpose(x,[1,0,2])),dtype=tf.float32,sequence_length=seq_length)
      
      lstm_out = tf.transpose(tf.stack(self.output),[1,0,2])
      
      if self.reuse is None:
        self.trainable_weights = vs.global_variables()
    self.reuse = True
    
    return lstm_out,last_state
      
    
class BiLSTM(object):
  '''
  LSTM layers using dynamic rnn
  '''
  def __init__(self,cell_size,num_layers=2,name='BiLSTM'):
    self.cell_size = cell_size
    self.num_layers = num_layers
    self.reuse = None
    self.trainable_weights = None
    self.name = name
  
  #x() equals to x.__call___()
  def __call__(self,x,seq_length=None,keep_prob=1.0):  #__call__ is very efficient when the state of instance changes frequently 
    with tf.variable_scope(self.name,reuse = self.reuse) as vs:
      
      fw_cell =tf.contrib.rnn.LSTMCell(self.cell_size,reuse=tf.get_variable_scope().reuse)
      bw_cell =tf.contrib.rnn.LSTMCell(self.cell_size,reuse=tf.get_variable_scope().reuse)
      
      fw_cell_drop = tf.contrib.rnn.DropoutWrapper(fw_cell,input_keep_prob=keep_prob)
      bw_cell_drop = tf.contrib.rnn.DropoutWrapper(bw_cell,input_keep_prob=keep_prob)
      
      
      if seq_length ==None:  #get the real sequence length (suppose that the padding are zeros)
        used = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2))
        seq_length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)
      
      lstm_out_bi,(output_state_fw,output_state_bw) =  tf.nn.bidirectional_dynamic_rnn(
                                                                       fw_cell_drop,
                                                                       bw_cell_drop,
                                                                       x,
                                                                       sequence_length=seq_length,
                                                                       dtype=tf.float32,
                                                                       time_major=False)
      lstm_out = tf.add(lstm_out_bi[0],lstm_out_bi[1])
      
      lstm_out_concat = tf.concat(lstm_out_bi,2)
      print 'lstm_out_concat:',lstm_out_concat
      #print 'output_state_fw:',output_state_fw
      #print 'output_state_fw:',output_state_fw[1]
      
      lstm_last_out = tf.concat([output_state_fw[1],output_state_bw][1],-1)
      #print 'lstm_out: ',lstm_out
      
      if self.reuse is None:
        self.trainable_weights = vs.global_variables()
        
    self.reuse =True
    return lstm_out_concat,lstm_out,lstm_last_out,seq_length

class FullyConnection(object):
  def __init__(self,output_size,name='FullyConnection'):
    self.output_size = output_size
    self.reuse = None
    self.trainable_weights = None
    self.name = name
    
  def __call__(self,inputs,activation_fn):
    with tf.variable_scope(self.name,reuse = self.reuse) as vs:
      out = tf.contrib.layers.fully_connected(inputs,self.output_size, activation_fn=activation_fn
                                           )
      if self.reuse is None:
        self.trainable_weights = vs.global_variables()
    self.reuse =True
    return out
  
class CRF(object):
  def __init__(self,output_size,name='CRF'):
    self.output_size = output_size
    self.reuse = None
    self.trainable_weights = None
    self.name = name
    
  def __call__(self,inputs,output_data,length):
    with tf.variable_scope(self.name,reuse =self.reuse) as vs:
      self.log_likelihood,self.transition_params  = tf.contrib.crf.crf_log_likelihood(inputs,output_data,length)
      self.loss = tf.reduce_mean(-self.log_likelihood)
      
      if self.reuse == None:
        self.trainable_weights = vs.global_variables()
          
    self.reuse = True
    
    return self.transition_params,self.loss

def classification_loss(flag,labels,logits):
  #logits = tf.nn.softmax(logits)
  if flag == 'figer':
    loss = tf.losses.mean_pairwise_squared_error(labels,logits)
    #print 'figer loss:',loss
  elif flag =='huber':
    loss = tf.losses.huber_loss(labels,logits)
  elif flag == 'softmax':
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
  elif flag=='sigmoid':
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits)
  elif flag =='hinge':
    loss = tf.losses.hinge_loss(logits,labels)
  elif flag =='KDD':
    max_pos = tf.nn.top_k(tf.multiply(labels,logits)).values
    max_neg = tf.nn.top_k(tf.multiply(1.0-labels,logits)).values
    loss = tf.nn.relu(1-(max_pos - max_neg))
    #print 'loss:',loss
  elif flag =='BBN':
    '''
    @貌似这个样子才符合EACL 2017提出来的东西哈！
    '''
    pos = tf.multiply(labels,tf.nn.relu(1-tf.multiply(labels,logits)))
    neg = tf.multiply(1.0-labels,tf.nn.relu(1+tf.multiply(1.0-labels,logits)))
    loss = tf.reduce_sum(pos + neg,-1)
    print 'loss:',loss
  elif flag == 'AAAI2017':
    labels = labels
    logits = logits
    loss = tf.multiply(labels,tf.scalar_mul(-1.0,tf.log(logits)))
    loss = tf.reduce_sum(loss,-1)
    #loss = tf.multiply(1.0/tf.reduce_sum(labels,-1),loss)
    return loss
  elif flag == 'AAAI2018':
    labels = labels
    logits = logits
    loss = tf.multiply(labels,tf.scalar_mul(-1.0,tf.log(logits))) + tf.multiply(1.0-labels,tf.scalar_mul(-1.0,tf.log(1.0-logits)))
    loss = tf.reduce_sum(loss,-1)
  else:
    loss = tf.losses.softmax_cross_entropy(labels,logits)  #must one-hot entropy
    
  return loss