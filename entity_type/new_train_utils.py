# -*- coding: utf-8 -*-

import numpy as np
from utils import genEntCtxMask_new,gen_pos_neg_tag
from hier_train_utils import mlmc_score,level_score,get_error
import random
import cPickle
import tensorflow as tf

pad_sequence = tf.contrib.keras.preprocessing.sequence.pad_sequences

def pad_odd(data):
  return np.concatenate((data,[data[-1]])) 
def getLinkingRet(tag,model_args,feed_data):
  sess = model_args.sess
  model = model_args.model
  args = model_args.args
  
  loss_type_list = []
  all_pred = []
  all_target = []
  all_wrong=0.0
  partial_wrong=0.0
  pred_fine=0.0
  pred_coarse= 0.0
  sample_size = len(feed_data)
  for i in range(sample_size):
    test_entment_mask,test_entment_chars,test_entment_chars_lent,test_sentence,test_tag = feed_data[i]
    
    test_out = test_tag
    
    test_input,\
        test_entMentIndex,entMent_length,\
        test_entCtxLeft_Index,ctxLeft_lent,pos2,test_entCtxRight_Index,ctxRight_lent,pos3 =  genEntCtxMask_new(args,args.batch_size,test_entment_mask,test_sentence)
    
    ent_rel_index = np.array([0]*len(test_out),np.int32)
    
    if len(test_out) != 2000:
      test_input = test_input[:len(test_out),:]
    
    feed_dict =  {
                    model.input_data:test_input,
                    model.entment_chars:test_entment_chars, 
                    model.entment_chars_lent:test_entment_chars_lent,
                    model.ent_rel_index:ent_rel_index,
                    model.output_data:test_out,
                    model.entMentIndex:test_entMentIndex,
                    model.entCtxLeftIndex:test_entCtxLeft_Index,
                    model.entCtxRightIndex:test_entCtxRight_Index,
                    model.entMentLent:entMent_length,
                    model.ctxLeftLent:ctxLeft_lent,
                    model.ctxRightLent:ctxRight_lent, 
                    model.is_training:False,
                    model.keep_prob:1}
    
    
    '''
    if args.model_type =='PengModel':
      train_sents = pad_sequence(test_sentence,args.sentence_length,padding='post',
                                     truncating='post',value=0)
      
      train_pos = []
      for i in range(len(test_out)):
        ent_index = map(int,test_entment_mask[i])
        ent_train = test_sentence[i]
        
        pos = []
        for i in range(len(ent_train)):
          if i < ent_index[1]:
            pos.append(i-int(ent_index[1])+args.sentence_length)
          elif i>=ent_index[1] and i<ent_index[2]:
            pos.append(0)
          else:
            pos.append(i-int(ent_index[2])+args.sentence_length)
        train_pos.append(pos)
        
      
      train_pos = pad_sequence(train_pos,args.sentence_length,padding='post',
                                 truncating='post',value=0)

      feed_dict[model.input_sents]=train_sents
      feed_dict[model.input_positions]=train_pos'''
    if 'ComplEx' in args.model_type:
      child,pos_neg,pos_neg_tag=model_args.type_tree.gen_struct_pair()
      feed_dict[model.complEx_type]=child
      feed_dict[model.complEx_pos_neg_type]=pos_neg
      feed_dict[model.complEx_pos_neg_type_tag]=pos_neg_tag
      if model_args.type_tree.hieral_layer==2:
        pos_types,pos_mask_l1,pos_mask_l2,neg_types = gen_pos_neg_tag(model_args,test_out)
      else:
        pos_types,pos_mask_l1,pos_mask_l2,pos_mask_l3,neg_types = gen_pos_neg_tag(model_args,test_out)
          
      feed_dict[model.pos_types]=pos_types
      feed_dict[model.neg_types]=neg_types
      feed_dict[model.pos_mask_l1]=pos_mask_l1
      feed_dict[model.pos_mask_l2]=pos_mask_l2
      
      if model_args.type_tree.hieral_layer==3:
        feed_dict[model.pos_mask_l3]=pos_mask_l3
            
    if args.model_type == 'DenoiseAbhishekModel':
      clean = []
      
      for train_i in range(len(test_out)):
        train_i_type_list = test_out[train_i]
        clean.append(model_args.type_tree.is_noise(train_i_type_list))
      feed_dict[model.clean] =clean
          
    loss_type,pred = sess.run([model.loss,model.prediction],feed_dict)
    loss_type_list.append(loss_type)
    
    pred_re,pred_score,target_re = model_args.type_tree.getResult(model_args.args.model_type,pred,test_out,args.threshold)
    
    all_pred = all_pred + list(pred_re)
    all_target = all_target + list(target_re)
    if tag=='testb':
      for sid in range(len(pred_re)):
        if set(pred_re[sid])!=set(target_re[sid]):
          sid_all_wrong,sid_partial_wrong,sid_pred_fine,sid_pred_coarse = get_error(target_re[sid],pred_re[sid])
          all_wrong += sid_all_wrong
          partial_wrong += sid_partial_wrong
          pred_fine += sid_pred_fine
          pred_coarse += sid_pred_coarse
  
  f1_strict,f1_macro,f1_micro = mlmc_score(model_args,all_pred,all_target)
  
  all_test = len(all_pred)
  
  all_wrong_err=all_wrong/all_test
  partial_wrong_err=partial_wrong/all_test
  pred_fine_err=pred_fine/all_test
  pred_coarse_err=pred_coarse/all_test
  if tag == 'testb':
    print(all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err)
    level_score(model_args,all_pred,all_target)
    
  if tag in 'testb':
    return all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err,np.sum(loss_type_list),f1_strict,f1_macro,f1_micro
  else:
    return np.sum(loss_type_list),f1_strict,f1_macro,f1_micro
         
         
def get_val(model_args,modesaveDataset,maximum,maximum_maF1,maximum_miF1,min_type_loss):
  sess = model_args.sess
  model = model_args.model
  args = model_args.args
  val_loss_type,val_f1_strict,val_f1_macro,val_f1_micro= getLinkingRet('testa',model_args,model_args.valid_data)
 
  #print("testa: loss_type:%.3f F1_s:%.3f f1_ma:%.3f f1_mi:%.3f"  %(val_loss_type,val_f1_strict,val_f1_macro,val_f1_micro))
  flag = True
  #val_loss_type <= min_type_loss and 
  flag =  val_f1_strict >= maximum #val_loss_type <= min_type_loss and val_f1_strict >= maximum
  
  if args.datasets=='OntoNotes':
    if model_args.args.model_type=='ShimaokeModel' or model_args.args.model_type=='PengModel':   
      flag = val_loss_type <min_type_loss
    
  if flag:
    maximum = val_f1_strict
    maximum_maF1 = val_f1_macro
    maximum_miF1 = val_f1_micro
    min_type_loss = val_loss_type
    
    #may cause something very strang...
    model.save(sess,args.restore,modesaveDataset,model.tvars) #optimize in the dev file!
    
    all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err,\
      test_loss_type,test_f1_strict,test_f1_macro,test_f1_micro = get_test_from_val(model_args,modesaveDataset)
      
    params = {'test_loss_type':test_loss_type,'test_f1_strict':test_f1_strict,
              'test_f1_macro':test_f1_macro,'test_f1_micro':test_f1_micro,
              'val_loss_type':val_loss_type,'val_f1_strict':val_f1_strict,
              'val_f1_macro':val_f1_macro,'val_f1_micro':val_f1_micro,
              'all_wrong_err':all_wrong_err,'partial_wrong_err':partial_wrong_err,
              'pred_fine_err':pred_fine_err,'pred_coarse_err':pred_coarse_err
              }
    
    cPickle.dump(params,open(args.dir_path + args.datasets+'/'+args.version_no+'/result/'+'opt.p_'+modesaveDataset,'wb'))

    print('----------------------------')
    print("testa: loss_type:%.3f F1_s:%.3f f1_ma:%.3f f1_mi:%.3f"  %(val_loss_type,val_f1_strict,val_f1_macro,val_f1_micro))
    print("testb: loss_type:%.3f F1_s:%.3f f1_ma:%.3f f1_mi:%.3f"  %(test_loss_type,test_f1_strict,test_f1_macro,test_f1_micro))

    print('------------------------------------------------')
  
    
  return maximum,maximum_maF1,maximum_miF1,min_type_loss

def get_test_from_val(model_args,modesaveDataset):
  args = model_args.args
  
  if args.datasets in ['OntoNotes','BBN','Wiki']:
    all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err,loss_type,f1_strict,f1_macro,f1_micro= getLinkingRet('testb',model_args,model_args.test_data)
   
  return all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err,loss_type,f1_strict,f1_macro,f1_micro
  
def print_ret(model_args,target,pred):
  target_ids =  list(np.nonzero(target)[0])
  target_type = []
  for key in target_ids:
    target_type.append(model_args.type_tree.id2type[key])
  print(target_ids, target_type)
  
  top_5 = list(np.argsort(-1*pred)[:10])
  pred_top_5 = []
  for key in top_5:
    pred_top_5.append([pred[key],model_args.type_tree.id2type[key]])
  print(top_5, pred_top_5)
  
  
#First version
#def gen_pos_neg_tag(model_args,data_tag_out):
#  args = model_args.args
#  
#  type_set = set(range(args.class_size))
#  sample_size = len(data_tag_out)
#  pos_tag_list = []
#  pos_mask_list = []
#  neg_tag_list = []
#  for i in range(sample_size):
#    complete_pos_tag = list(np.nonzero(data_tag_out[i])[0])
#    
#    type_lent = len(complete_pos_tag)
#    #we need to attention that what we do there!
#    if type_lent == args.max_pos_type:
#      pos_mask = [1.0]*args.max_pos_type
#      pos_tag = complete_pos_tag
#    else:
#      pos_tag = complete_pos_tag + [args.class_size]*(args.max_pos_type-type_lent)
#      pos_mask = [1.0]*type_lent +[0.0]*(args.max_pos_type-type_lent)
#      
#    assert(len(pos_tag)==args.max_pos_type)
#    assert(len(pos_mask)==args.max_pos_type)
#    pos_tag_list.append(pos_tag)
#    pos_mask_list.append(pos_mask)
#    neg_tag_list.append(random.sample(list(type_set-set(complete_pos_tag)),args.max_neg_type))
#  return np.asarray(pos_tag_list,np.int32),np.asarray(pos_mask_list,np.float32),np.array(neg_tag_list,np.int32) 

#def gen_type_pair(model_args,data_tag_out):
#  args = model_args.args
#  type_set = set(range(args.class_size))
#  sample_size = len(data_tag_out)
#  type_pair_list = []
#  type_cor_score_list = []
#  for i in range(sample_size):
#    complete_pos_tag = list(np.nonzero(data_tag_out[i])[0])
#    
#    pos_tag = random.choice(complete_pos_tag)
#    neg_tag = random.choice(list(type_set-set([pos_tag])))  #do not compute score with itself...
#    type_pair_list.append([pos_tag,neg_tag])
#    
#    try:
#      correlation = model_args.type_correlation[pos_tag][neg_tag]
#    except:
#      correlation = 0.0
#    type_cor_score_list.append(correlation)
#  #print type_cor_score_list[:5] 
#  return np.array(type_pair_list,np.int32),np.array(type_cor_score_list,np.float32)