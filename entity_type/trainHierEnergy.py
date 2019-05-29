# -*- coding: utf-8 -*-

import sys
sys.path.append('utils')
sys.path.append("embedding")
import numpy as np
import tensorflow as tf
import cPickle
import time
from utils import genEntCtxMask_new,gen_pos_neg_tag
from hier_train_utils import get_val,get_test_from_val,mlmc_score
import model_args

def main(_):
  args = model_args.args
  model = model_args.model
  
  #summary = tf.summary.merge_all()
  sess = model_args.sess
  learning_rate = args.learning_rate
  #summary_writer = tf.summary.FileWriter(args.log_dir)
  
  if args.artifical_noise_weight!=0.0:
    artifical_ent_dict = cPickle.load(open(args.dir_path + args.datasets+'/'+args.version_no+ '/generated/'+'artifical_ent_dict.p'+'_'
                        +str(args.artifical_noise_weight),'rb'))
    
    aritifical_all_ent_dict = cPickle.load(open(args.dir_path + args.datasets+'/'+args.version_no+'/generated/'+'aritifical_all_ent_dict.p'+'_'
                        +str(args.artifical_noise_weight),'rb'))
    
    model_args.type_tree.artifical_ent_dict = artifical_ent_dict
  
  modesaveDataset ='_'.join(['artifical',str(args.artifical_noise_weight),
                             args.datasets,
                             args.model_type,
                             str(args.learning_rate),
                             str(args.rnn_size),str(args.attention_hidden_size),
                             str(args.type_dim) ,str(args.margin),'sl2',str(args.l2_loss_w),
                             str(args.max_neg_type),str(args.max_pos_type),
                              str(args.is_add_fnode)
                      ])
  print('start to build seqLSTM')
  
  if args.is_test ==True or args.init_epoch!=0:
    if model.load(sess,args.restore,modesaveDataset,model.tvars):
      print( "[*] "+modesaveDataset +"is loaded...")
    else:
      print( 'no checkpoint: '+ modesaveDataset)
      
  #to determine the test or train
  if args.is_test:
    min_type_loss = sys.maxint
    maximum=0;maximum_maF1=0;maximum_miF1=0
    maximum,maximum_maF1,maximum_miF1,min_type_loss = get_val(model_args,modesaveDataset,maximum,maximum_maF1,maximum_miF1,min_type_loss)
    
#    loss_type,f1_strict,f1_macro,f1_micro = get_test_from_val(model_args,modesaveDataset)
#    print("testb: loss_type:%.3f F1_s:%.3f f1_ma:%.3f f1_mi:%.3f"  %(loss_type,f1_strict,f1_macro,f1_micro))
#    print( '--------------------------------')
#    params = {'test_loss_type':loss_type,'test_f1_strict':f1_strict,
#              'test_f1_macro':f1_macro,'test_f1_micro':f1_micro,
#              }
#    cPickle.dump(params,open(args.dir_path + args.datasets+'/'+args.version_no+'/result/'+'opt.p_'+modesaveDataset,'wb'))
  else:
    min_type_loss = sys.maxint
    maximum=0;maximum_maF1=0;maximum_miF1=0
    train_datas =list(model_args.train_collection.find({}))
    id_epoch = 0
    for i_epoch in range(args.epochs):
      print('epoch:',i_epoch)
      print('--------------------------')
      stime = time.time()
      new_id_epoch=0
      for train_entment_mask,\
      train_entment_chars,train_entment_chars_lent,\
      train_sentence_final,train_tag_final,\
      ent_rel_index in model_args.train_data_reader.get_shuffle_train_data(args.batch_size,
model_args.training_data_nums,train_datas ):
       
        train_input,train_entMentIndex,entMent_length,\
         train_entCtxLeft_Index,ctxLeft_lent,pos2,\
         train_entCtxRight_Index,ctxRight_lent,pos3 =  genEntCtxMask_new(args,args.batch_size,train_entment_mask, train_sentence_final)
        if args.artifical_noise_weight==0.0:
          train_out = np.asarray(train_tag_final,np.float32)
        else:
          new_train_out = []
          for ids in ent_rel_index:
            type_set  = aritifical_all_ent_dict[ids]
            new_train_out_i = [0] * args.class_size
            for ti in type_set:
              new_train_out_i[ti]=1
            new_train_out.append(new_train_out_i)
            
          train_out = np.asarray(new_train_out,np.float32)
        
        if model_args.type_tree.hieral_layer==2:
          pos_types,pos_mask_l1,pos_mask_l2,neg_types = gen_pos_neg_tag(model_args,train_out)
        else:
          pos_types,pos_mask_l1,pos_mask_l2,pos_mask_l3,neg_types = gen_pos_neg_tag(model_args,train_out)
        if len(train_out) != 2000:
          train_input = train_input[:len(train_out),:]
        new_id_epoch += 1
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
                           model.learning_rate:learning_rate,
                           model.keep_prob:0.5}
        
        if model_args.type_tree.hieral_layer==3:
          feed_in_dict[model.pos_mask_l3]=pos_mask_l3
        
        _,train_loss_type,train_loss_total,pred,a_norm= sess.run([model.train_pos_op,
                                                                  model.loss,
                                                                  model.total_loss,
                                                                  model.prediction,
                                                                  model.a_norm], feed_in_dict)
        if new_id_epoch % (args.iterateEpoch*5) == 0:
          pred_re,pred_score,train_out_re = model_args.type_tree.getResult(model_args.args.model_type,pred,train_out,args.threshold)
          
          f1_strict,f1_macro,f1_micro = mlmc_score(model_args,pred_re,train_out_re)
          print("time: %.2f sent_id: %d, train: loss_type:%.2f loss_total:%.2f F1_s:%.2f f1_ma:%.2f f1_mi:%.2f, lr:%.6f"  %(time.time()-stime,new_id_epoch*args.batch_size,train_loss_type,train_loss_total, f1_strict*0.01,f1_macro*0.01,f1_micro*0.01,learning_rate))
        if new_id_epoch % args.iterateEpoch == 0:
          maximum,maximum_maF1,maximum_miF1,min_type_loss = get_val(model_args,modesaveDataset,maximum,maximum_maF1,maximum_miF1,min_type_loss)
      maximum,maximum_maF1,maximum_miF1,min_type_loss = get_val(model_args,modesaveDataset,maximum,maximum_maF1,maximum_miF1,min_type_loss)
      
      all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err,\
      test_loss_type,test_f1_strict,test_f1_macro,test_f1_micro = get_test_from_val(model_args,modesaveDataset)
      
      print("testb: loss_type:%.3f F1_s:%.3f f1_ma:%.3f f1_mi:%.3f"  %(test_loss_type,test_f1_strict*0.01,test_f1_macro*0.01,test_f1_micro*0.01))
    print('--------------------------------')
          
  model_args.client.close()
  sess.close()
 
if __name__=='__main__':
  tf.app.run()