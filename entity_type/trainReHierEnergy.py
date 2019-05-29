# -*- coding: utf-8 -*-
import sys
sys.path.append('utils')
import sys
sys.path.append("embedding")
import numpy as np
import tensorflow as tf
import cPickle
from utils import genEntCtxMask_new
from evals import MentTrieTree
from hier_train_utils import get_val,get_test_from_val,gen_pos_neg_tag
import model_args
from tqdm import tqdm

args = model_args.args
summary = tf.summary.merge_all()
sess = model_args.sess
summary_writer = tf.summary.FileWriter(args.log_dir)
  
if args.artifical_noise_weight!=0.0:
  artifical_ent_dict = cPickle.load(open(args.dir_path + args.datasets+'/'+args.version_no+ '/generated/'+'artifical_ent_dict.p'+'_'
                      +str(args.artifical_noise_weight),'rb'))
  
  aritifical_all_ent_dict = cPickle.load(open(args.dir_path + args.datasets+'/'+args.version_no+'/generated/'+'aritifical_all_ent_dict.p'+'_'
                      +str(args.artifical_noise_weight),'rb'))
  
  model_args.type_tree.artifical_ent_dict = artifical_ent_dict
  
def main(_):
  
  modesaveDataset_old = '_'.join(['artifical',str(args.artifical_noise_weight),args.datasets,
                            args.model_type, str(args.learning_rate),
                             str(args.rnn_size),str(args.attention_hidden_size),
                             str(args.type_dim) ,str(args.margin),'sl2',str(args.l2_loss_w),
                             str(args.max_neg_type),str(args.max_pos_type),
                             str(args.is_add_fnode),
                             'r',str(args.iterate_num-1),str(args.filter_threshold)
                    ])
    
  modesaveDataset = '_'.join(['artifical',str(args.artifical_noise_weight),args.datasets,
                            args.model_type,str(args.learning_rate),
                             str(args.rnn_size),str(args.attention_hidden_size),
                             str(args.type_dim) ,str(args.margin),'sl2',str(args.l2_loss_w),
                             str(args.max_neg_type),str(args.max_pos_type),
                             str(args.is_add_fnode),
                             'r',str(args.iterate_num),str(args.filter_threshold)
                    ])
  
  if args.iterate_num==1:
    load_modesaveDataset = '_'.join(['artifical',str(args.artifical_noise_weight),args.datasets,args.model_type,str(args.learning_rate),
                               str(args.rnn_size),str(args.attention_hidden_size),
                               str(args.type_dim) ,str(args.margin),'sl2',str(args.l2_loss_w),
                               str(args.max_neg_type),str(args.max_pos_type),
                               str(args.is_add_fnode)
                      ])
  else:
    load_modesaveDataset = modesaveDataset_old
   
  
  print('start to build seqLSTM')
  model = model_args.model
  sess = model_args.sess

  #to determine the test or train
  if args.is_test:
    if model.load(sess,args.restore,modesaveDataset,model.tvars):
      print("[*] "+modesaveDataset +"is loaded...")
    else:
      print('no checkpoint: '+ modesaveDataset)
    
    min_type_loss = sys.maxint
    maximum=0;maximum_maF1=0;maximum_miF1=0
    maximum,maximum_maF1,maximum_miF1,min_type_loss = get_val(model_args,modesaveDataset,maximum,maximum_maF1,maximum_miF1,min_type_loss)
  else:
    if model.load(sess,args.restore,load_modesaveDataset,model.tvars):
      print("[*] "+load_modesaveDataset +"is loaded...")
    else:
      print('no checkpoint: '+ load_modesaveDataset)
    
    last_new_gen_train_out = cPickle.load(open(args.dir_path + args.datasets+'/'+args.version_no+'/iter_generated/'+'new_gen_train_out.p_'+modesaveDataset_old,'rb'))

    min_type_loss = sys.maxint
    maximum=0;maximum_maF1=0;maximum_miF1=0
    train_datas =list(model_args.train_collection.find({}))
    
    for i_epoch in range(args.epochs):
      print('epoch',i_epoch)
      id_epoch = 0
      print('--------------------------')
      for train_entment_mask,\
      train_entment_chars,train_entment_chars_lent,\
      train_sentence_final,train_tag_final,\
      ent_rel_index in tqdm(model_args.train_data_reader.get_shuffle_train_data(args.batch_size,
                                                                           model_args.training_data_nums,
                                                                           train_datas)):     
        train_input,train_entMentIndex,entMent_length,\
         train_entCtxLeft_Index,ctxLeft_lent,pos2,\
         train_entCtxRight_Index,ctxRight_lent,pos3 =  genEntCtxMask_new(args,args.batch_size,train_entment_mask,
                                                                        train_sentence_final)
        
        new_train_out=[]
        for i in range(len(ent_rel_index)):
          ids = ent_rel_index[i]
          type_set  = last_new_gen_train_out[ids]
          new_train_out_i = [0] * args.class_size
          for ti in type_set:
            new_train_out_i[ti]=1
          new_train_out.append(new_train_out_i)
        
        
        train_out = np.asarray(new_train_out,np.float32)
        if len(train_out)==0:
          continue
        
        
        #we neeed to make sure the deleted type need to be added into negative types...
        
        if model_args.type_tree.hieral_layer==2:
          pos_types,pos_mask_l1,pos_mask_l2,neg_types = gen_pos_neg_tag(model_args,train_out)
        else:
          pos_types,pos_mask_l1,pos_mask_l2,pos_mask_l3,neg_types = gen_pos_neg_tag(model_args,train_out)
        
        if len(train_out) != 2000:
          train_input = train_input[:len(train_out),:]
          
        #to avoid the overfitting...
        feed_in_dict= {model.input_data:train_input,
                           model.entment_chars:train_entment_chars, model.entment_chars_lent:train_entment_chars_lent,
                           model.pos_types:pos_types,
                           model.neg_types:neg_types,
                           model.ent_rel_index:ent_rel_index,
                           model.output_data:train_out,
                           model.pos_mask_l1:pos_mask_l1,
                           model.pos_mask_l2:pos_mask_l2,
                           model.entMentIndex:train_entMentIndex,
                           model.entCtxLeftIndex:train_entCtxLeft_Index,
                           model.entCtxRightIndex:train_entCtxRight_Index,
                           model.entMentLent:entMent_length,
                           model.ctxLeftLent:ctxLeft_lent,model.ctxRightLent:ctxRight_lent,
                           model.learning_rate:args.learning_rate,
                           model.keep_prob:1.0}  #to improve the the context support?
        
        if model_args.type_tree.hieral_layer==3:
          feed_in_dict[model.pos_mask_l3]=pos_mask_l3
        
        _,l2_loss,train_loss_type,train_loss_total,\
           pred,a_norm= sess.run([model.train_pos_op,model.l2_loss,model.loss,model.total_loss,
                                  model.prediction,model.a_norm],feed_in_dict
                          )

        id_epoch += 1
        if id_epoch % args.iterateEpoch == 0:   #we utilize the batch to 
          maximum,maximum_maF1,maximum_miF1,min_type_loss = get_val(model_args,modesaveDataset,maximum,maximum_maF1,maximum_miF1,min_type_loss)
        if args.datasets=='Wiki' and id_epoch % 500==0:  #to avoid the overfiting
          break
      maximum,maximum_maF1,maximum_miF1,min_type_loss = get_val(model_args,modesaveDataset,maximum,maximum_maF1,maximum_miF1,min_type_loss)
          
  model_args.client.close()
  sess.close()
 
if __name__=='__main__':
  tf.app.run()