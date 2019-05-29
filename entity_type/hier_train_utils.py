# -*- coding: utf-8 -*-

import numpy as np
import cPickle
from utils import genEntCtxMask_new,gen_pos_neg_tag
from evals import MentTrieTree

def level_types(model_args,type_vector,is_set=False): #[class_size]
  #only contain the leaf types...
  if is_set==False:  
    target_type_list = np.nonzero(type_vector)[0]
  else:
    target_type_list=type_vector
    
  tree = MentTrieTree()
  type_name_list= []
  for  ti in target_type_list:
    tname = model_args.type_tree.id2type[ti]
    tree.add(tname)
    type_name_list.append(tname)
  
  target_type_set=set()
  for tname in type_name_list:  #only consider the leaf nodes
    if tree.is_leaf(tname):
      
      if tname in model_args.type_tree.l1_type2id:
        target_type_set.add(tname+'_t1')
      elif tname in model_args.type_tree.l2_type2id:
        target_type_set.add(tname+'_t2')
      else:
        target_type_set.add(tname+'_t3')
  
  return target_type_set

    
def level_score(model_args,predict,target):
  samples = len(predict)
  
  t_level=[0.0,0.0,0.0]  #target
  p_level=[0.0,0.0,0.0]  #predict
  j_level=[0.0,0.0,0.0]  #joint
  for i in range(samples):
    target_i = set(target[i])
    predict_i = set(predict[i])
    
    t_level_set = level_types(model_args,target_i,is_set=True)
    p_level_set = level_types(model_args,predict_i,is_set=True)
    
    p_t_joint = list(t_level_set & p_level_set)
    p_t_joint_lent = len(p_t_joint)
    
    for ti in t_level_set:
      if '_t1' in ti:
        t_level[0]+=1
      elif '_t2' in ti:
        t_level[1]+=1
      else:
        t_level[2]+=1
    '''
    @对不同层进行一下处理啦~~
    '''
    if p_t_joint_lent!=0:
      for ti in p_t_joint:
        if '_t1' in ti:
          j_level[0]+=1
        elif '_t2' in ti:
          j_level[1]+=1
        else:
          j_level[2]+=1
    
    for ti in p_level_set:
      if '_t1' in ti:
        p_level[0]+=1
      elif '_t2' in ti:
        p_level[1]+=1
      else:
        p_level[2]+=1
      
  print(t_level)
  print(p_level)
  print(j_level)
  for i in range(3):
    if t_level[i]!=0 and p_level[i]!=0:
      p = j_level[i]/p_level[i]
      r = j_level[i]/t_level[i]
      if p+r ==0:
        f1 = 0
      else:
        f1=2*p*r/(p+r)
      print(p,r,f1)
  print('-----------')

def mlmc_score(model_args,predict,target):
  samples = len(predict)
  
  right = 0.0
  sample_precision=[]
  sample_recall = []
  
  sample_joint_lent =0.0
  sample_pred_lent = 0.0
  sample_target_lent = 0.0
  for i in range(samples):
    target_i = set(target[i])
    predict_i = set(predict[i])
    
    #target_i = level_types(model_args,target[i],is_set=True)
    #predict_i = level_types(model_args,predict[i],is_set=True)
    
    
    t_lent = len(target_i) * 1.0
    p_lent = len(predict_i) *1.0
    
    if target_i == predict_i:
      right += 1
    
    p_t_joint = list(target_i & predict_i)
    p_t_joint_lent = len(p_t_joint)
    
    if p_lent==0:
      sample_precision.append(0)
    else:
      sample_precision.append(p_t_joint_lent/p_lent)
      
    sample_recall.append(p_t_joint_lent/t_lent)
    
    sample_joint_lent += p_t_joint_lent
    sample_target_lent += t_lent
    sample_pred_lent += p_lent
  
  #f1_strict 
  f1_strict= right/samples
  
  #macro
  p_macro = np.average(sample_precision)
  r_macro = np.average(sample_recall)
  
  if p_macro + r_macro==0:
    f1_macro=0.0
  else:
    f1_macro = 2*p_macro*r_macro/(p_macro+r_macro)
  
  #micro
  if sample_pred_lent==0:
    p_micro=0.0
  else:
    p_micro = sample_joint_lent/sample_pred_lent
    
  r_micro = sample_joint_lent/sample_target_lent
  
  if p_micro + r_micro==0:
    f1_micro=0
  else:
    f1_micro = 2*p_micro*r_micro/(p_micro+r_micro)
  
  return  f1_strict*100,f1_macro*100,f1_micro*100

def pad_odd(data):
  return np.concatenate((data,[data[-1]])) 

def get_error(target_re,pred_re):
  target_type_set = set(target_re)
  pred_type_set = set(pred_re)
  
  all_wrong = 0
  partial_wrong=0
  pred_fine=0
  pred_coarse=0
  
  if target_type_set & pred_type_set == set():
    all_wrong = 1
  else:
    target_type_lent =len(target_type_set)
    pred_type_lent = len(pred_type_set)
    
    if target_type_lent > pred_type_lent:
      pred_coarse =1
    elif target_type_lent < pred_type_lent:
      pred_fine = 1
    else:
      partial_wrong = 1
  
  return all_wrong,partial_wrong,pred_fine,pred_coarse

def getLinkingRet(tag,model_args,feed_data):
  loss_type_list = []
  model = model_args.model
  sess = model_args.sess
  all_wrong=0.0
  partial_wrong=0.0
  pred_fine=0.0
  pred_coarse= 0.0
  sample_size = len(feed_data)
  all_pred = []
  all_target = []
  
  for i in range(sample_size):
    test_entment_mask,test_entment_chars,test_entment_chars_lent,test_sentence,test_tag = feed_data[i]
    
    test_out = test_tag
    
    test_input,\
        test_entMentIndex,entMent_length,\
        test_entCtxLeft_Index,ctxLeft_lent,pos2,test_entCtxRight_Index,ctxRight_lent,pos3 =  genEntCtxMask_new(model_args.args,model_args.args.batch_size,test_entment_mask,test_sentence)
        
    ent_rel_index = np.array([0]*len(test_out),np.int32)
    
    if model_args.args.is_training == True:
      if model_args.type_tree.hieral_layer==2:  #may take a lot of time...
        pos_types,pos_mask_l1,pos_mask_l2,neg_types = gen_pos_neg_tag(model_args,test_out)
      else:
        pos_types,pos_mask_l1,pos_mask_l2,pos_mask_l3,neg_types = gen_pos_neg_tag(model_args,test_out)
        
      feed_in_dict= {
                      model.input_data:test_input,
                      model.entment_chars:test_entment_chars, 
                      model.entment_chars_lent:test_entment_chars_lent,
                      model.pos_types:pos_types,
                      model.neg_types:neg_types,
                      model.pos_mask_l2:pos_mask_l2,
                      model.pos_mask_l1:pos_mask_l1,
                      model.ent_rel_index:ent_rel_index,
                      model.output_data:test_out,
                      model.entMentIndex:test_entMentIndex,
                      model.entCtxLeftIndex:test_entCtxLeft_Index,
                      model.entCtxRightIndex:test_entCtxRight_Index,
                      model.entMentLent:entMent_length,
                      model.ctxLeftLent:ctxLeft_lent,
                      model.ctxRightLent:ctxRight_lent, 
                      model.learning_rate:1.0,
                      model.keep_prob:1}
      if model_args.type_tree.hieral_layer==3:
        feed_in_dict[model.pos_mask_l3]=pos_mask_l3
    else:
      feed_in_dict= {
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
                      model.keep_prob:1}
    #
    loss_type,pred = sess.run([model.total_loss,model.prediction],feed_in_dict )
    loss_type_list.append(loss_type)

    pred_re,pred_score,target_re = model_args.type_tree.getResult(model_args.args.model_type,pred,test_out,model_args.args.threshold)

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
          
  all_test = len(all_pred)
  #print('all_test:',all_test)
  all_wrong_err=all_wrong/all_test
  partial_wrong_err=partial_wrong/all_test
  pred_fine_err=pred_fine/all_test
  pred_coarse_err=pred_coarse/all_test
  if tag == 'testb':
    print(all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err)
    
    level_score(model_args,all_pred,all_target)
  
  f1_strict,f1_macro ,f1_micro = mlmc_score(model_args,all_pred,all_target)
  if tag == 'testb':
    return all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err,\
              np.sum(loss_type_list),f1_strict,f1_macro,f1_micro
  else:
    return np.sum(loss_type_list),f1_strict,f1_macro,f1_micro
         
         
def get_val(model_args,modesaveDataset,maximum,maximum_maF1,maximum_miF1,min_type_loss):
  model = model_args.model
  sess = model_args.sess
  args = model_args.args
  val_loss_type,val_f1_strict,val_f1_macro,val_f1_micro= getLinkingRet('testa',model_args,model_args.valid_data)

  flag = True
  
  flag = val_f1_macro >= maximum_maF1#(val_loss_type <= min_type_loss and val_f1_strict >= maximum and val_f1_macro >= maximum_maF1 and val_f1_micro >= maximum_miF1)
  
  if flag:
    maximum = val_f1_strict 
    maximum_maF1 = val_f1_macro
    maximum_miF1 = val_f1_micro
    min_type_loss = val_loss_type
    model.save(sess,model_args.args.restore,modesaveDataset,model.tvars) #optimize in the dev file!
    print('----------------------------')
    print("testa: loss_type:%.3f F1_s:%.3f f1_ma:%.3f f1_mi:%.3f"  %(val_loss_type,val_f1_strict*0.01,val_f1_macro*0.01,val_f1_micro*0.01))
    all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err,\
      test_loss_type,test_f1_strict,test_f1_macro,test_f1_micro = get_test_from_val(model_args,modesaveDataset)
      
    print("testb: loss_type:%.3f F1_s:%.3f f1_ma:%.3f f1_mi:%.3f"  %(test_loss_type,test_f1_strict*0.01,test_f1_macro*0.01,test_f1_micro*0.01))
    print('--------------------------------')
    params = {'test_loss_type':test_loss_type,'test_f1_strict':test_f1_strict,
              'test_f1_macro':test_f1_macro,'test_f1_micro':test_f1_micro,
              'val_loss_type':val_loss_type,'val_f1_strict':val_f1_strict,
              'val_f1_macro':val_f1_macro,'val_f1_micro':val_f1_micro,
              'all_wrong_err':all_wrong_err,'partial_wrong_err':partial_wrong_err,
              'pred_fine_err':pred_fine_err,'pred_coarse_err':pred_coarse_err
              }
    
    cPickle.dump(params,open(args.dir_path + args.datasets+'/'+args.version_no+'/result/'+'opt.p_'+modesaveDataset,'wb'))
    print('------------------------------------------------')
  return maximum,maximum_maF1,maximum_miF1,min_type_loss

def get_test_from_val(model_args,modesaveDataset):
  all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err,loss_type,f1_strict,f1_macro,f1_micro= getLinkingRet('testb',model_args,model_args.test_data)
  return all_wrong_err,partial_wrong_err,pred_fine_err,pred_coarse_err,loss_type,f1_strict,f1_macro,f1_micro