# -*- coding: utf-8 -*-

import sys
sys.path.append('utils')
import numpy as np
import tensorflow as tf
import cPickle
from hier_train_utils import mlmc_score
from new_train_utils import get_val,get_test_from_val
from utils import genEntCtxMask_new,gen_pos_neg_tag
from evals import MentTrieTree
import model_args
import time

args = model_args.args
pad_sequence = tf.contrib.keras.preprocessing.sequence.pad_sequences
if args.artifical_noise_weight!=0.0:
  artifical_ent_dict = cPickle.load(open(args.dir_path + args.datasets+'/'+args.version_no+ '/generated/'+'artifical_ent_dict.p'+'_'
                      +str(args.artifical_noise_weight),'rb'))
  
  aritifical_all_ent_dict = cPickle.load(open(args.dir_path + args.datasets+'/'+args.version_no+'/generated/'+'aritifical_all_ent_dict.p'+'_'
                      +str(args.artifical_noise_weight),'rb'))
  
  model_args.type_tree.artifical_ent_dict = artifical_ent_dict

def main(_):
  print('max_pos_type:',args.max_pos_type)
  modesaveDataset ='_'.join(['artifical',str(args.artifical_noise_weight),args.datasets,args.model_type,
                             str(args.use_clean),
                             str(args.learning_rate),
                             str(args.rnn_size),str(args.attention_hidden_size),
                             str(args.type_dim) ,str(args.margin),
                             str(args.max_neg_type),
                             str(args.max_pos_type),
                             str(args.alpha)
                      ])
  
  print('start to build seqLSTM')
  model = model_args.model
  sess = model_args.sess
  
  if args.init_epoch!=0:
    if model.load(sess,args.restore,modesaveDataset,model.tvars):
      print( "[*] "+modesaveDataset +"is loaded...")
    else:
      print( 'no checkpoint: '+ modesaveDataset)
    
  #to determine the test or train
  if args.is_test:
    if model.load(sess,args.restore,modesaveDataset,model.tvars):
      print( "[*] "+modesaveDataset +"is loaded...")
    else:
      print( 'no checkpoint: '+ modesaveDataset)
    min_type_loss = sys.maxint
    maximum=0;maximum_maF1=0;maximum_miF1=0
    maximum,maximum_maF1,maximum_miF1,min_type_loss =     get_val(model_args,modesaveDataset,maximum,maximum_maF1,maximum_miF1,min_type_loss)
  else:
    min_type_loss = sys.maxint
    maximum=0;maximum_maF1=0;maximum_miF1=0
    train_datas =list(model_args.train_collection.find({}))
    for i_epoch in range(args.epochs):
      id_epoch = 0
      print('epoch',i_epoch)
      stime = time.time()
      print('--------------------------')
      for train_entment_mask,\
      train_entment_chars,train_entment_chars_lent,\
      train_sentence_final,train_tag_final,\
      ent_rel_index in model_args.train_data_reader.get_shuffle_train_data(args.batch_size,
                                                                             model_args.training_data_nums,train_datas
                                                                             ):
        train_input,train_entMentIndex,entMent_length,\
         train_entCtxLeft_Index,ctxLeft_lent,pos2,\
         train_entCtxRight_Index,ctxRight_lent,pos3 =  genEntCtxMask_new(args,args.batch_size,
                                                                         train_entment_mask,
                                                                        train_sentence_final)
        
        if args.artifical_noise_weight==0.0:
          train_out_path = np.asarray(train_tag_final,np.float32)
        else:
          train_out_path = []
          for ids in ent_rel_index:
            type_set  = aritifical_all_ent_dict[ids]
            new_train_out_i = [0] * args.class_size
            for ti in type_set:
              new_train_out_i[ti]=1
            train_out_path.append(new_train_out_i)
          train_out_path = np.asarray(train_out_path,np.float32)
        
        if args.model_type =='PengModel':
          train_out = []
          
          for i in range(len(train_out_path)):
            tmp_out = np.zeros(args.class_size)
            enti_all_types = np.nonzero(train_out_path[i])[0]
            tree = MentTrieTree()
            type_name_list= []
            for  ti in enti_all_types:
              tname = model.trieTree.id2type[ti]
              tree.add(tname)
              type_name_list.append(tname)
              
            for tname in type_name_list:
              if tree.is_leaf(tname):
                tname_id = model.trieTree.type2id[tname]
                tmp_out[tname_id]=1
            train_out.append(tmp_out)
          train_out = np.array(train_out,np.float32)
        else:
          train_out = train_out_path
          
        feed_dict = {model.input_data:train_input,
                     model.entment_chars:train_entment_chars, model.entment_chars_lent:train_entment_chars_lent,
                     model.ent_rel_index:ent_rel_index,
                     model.output_data:train_out,
                     model.entMentIndex:train_entMentIndex,
                     model.entCtxLeftIndex:train_entCtxLeft_Index,
                     model.entCtxRightIndex:train_entCtxRight_Index,
                     model.entMentLent:entMent_length,
                     model.ctxLeftLent:ctxLeft_lent,
                     model.ctxRightLent:ctxRight_lent,
                     model.is_training:True,
                     model.keep_prob:args.keep_prob}
        
        if 'ComplEx' in args.model_type:
          child,pos_neg,pos_neg_tag=model_args.type_tree.gen_struct_pair()
          feed_dict[model.complEx_type]=child
          feed_dict[model.complEx_pos_neg_type]=pos_neg
          feed_dict[model.complEx_pos_neg_type_tag]=pos_neg_tag
          
          if model_args.type_tree.hieral_layer==2:
            pos_types,pos_mask_l1,pos_mask_l2,neg_types = gen_pos_neg_tag(model_args,train_out)
          else:
            pos_types,pos_mask_l1,pos_mask_l2,pos_mask_l3,neg_types = gen_pos_neg_tag(model_args,train_out)
          
          feed_dict[model.pos_types]=pos_types
          feed_dict[model.neg_types]=neg_types
          feed_dict[model.pos_mask_l1]=pos_mask_l1
          feed_dict[model.pos_mask_l2]=pos_mask_l2
                           
          if model_args.type_tree.hieral_layer==3:
            feed_dict[model.pos_mask_l3]=pos_mask_l3
        
        if args.model_type == 'DenoiseAbhishekModel':
          clean = []
          
          for train_i in range(len(train_out)):
            train_i_type_list = train_out[train_i]
            clean.append(model_args.type_tree.is_noise(train_i_type_list))
          feed_dict[model.clean] =clean
        
        id_epoch += 1
        if id_epoch % 10 ==0:
          _,train_loss_type,train_loss_total,pred= sess.run([model.train_pos_op,model.loss,model.total_loss,model.prediction],feed_dict)
           
          pred_re,pred_score,train_out_re = model_args.type_tree.getResult(model_args.args.model_type,pred,train_out,args.threshold)
          
          f1_strict,f1_macro,f1_micro = mlmc_score(model_args,pred_re,train_out_re)
          print("time: %.2f sent_id: %d, train: loss_type:%.2f loss_total:%.2f F1_s:%.2f f1_ma:%.2f f1_mi:%.2f, lr:%.6f"  %(time.time()-stime,id_epoch*args.batch_size,train_loss_type,train_loss_total, f1_strict*0.01,f1_macro*0.01,f1_micro*0.01,model_args.args.learning_rate))
          
        if id_epoch % args.iterateEpoch ==0:
          maximum,maximum_maF1,maximum_miF1,min_type_loss =get_val(model_args,modesaveDataset, maximum,maximum_maF1,maximum_miF1,min_type_loss)
      
      maximum,maximum_maF1,maximum_miF1,min_type_loss =get_val(model_args,modesaveDataset,
                                                               maximum,maximum_maF1,maximum_miF1,min_type_loss)
      all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err,\
      test_loss_type,test_f1_strict,test_f1_macro,test_f1_micro = get_test_from_val(model_args,modesaveDataset)
      
      print("testb: loss_type:%.3f F1_s:%.3f f1_ma:%.3f f1_mi:%.3f"  %(test_loss_type,test_f1_strict*0.01,test_f1_macro*0.01,test_f1_micro*0.01))
    print('--------------------------------')
    
  model_args.client.close()
  sess.close()
 
if __name__=='__main__':
  tf.app.run()