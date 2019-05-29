# -*- coding: utf-8 -*-

import numpy as np
import random
from gen_1_layer_hier_pos_type import gen_hier_type
from gen_2_layer_hier_pos_type import gen_hier_type_2

def gen_pos_neg_tag(model_args,data_tag_out):
  sample_size = len(data_tag_out)
  
  pos_tag_list = []
  pos_mask_l1_list=[]
  pos_mask_l2_list=[]
  pos_mask_list = []
  neg_tag_list = []
  for i in range(sample_size):
    complete_pos_tag = list(np.nonzero(data_tag_out[i])[0])
    #we need to generate the hierarichal 
    if model_args.type_tree.hieral_layer==2:
      neg_tag,pos_type,pos_type_mask_l1,pos_type_mask_l2 = gen_hier_type(model_args,complete_pos_tag)
      
      pos_tag_list.append(pos_type)
      pos_mask_l1_list.append(pos_type_mask_l1)
      pos_mask_list.append(pos_type_mask_l2)
      neg_tag_list.append(neg_tag)
    elif model_args.type_tree.hieral_layer==3:
      neg_tag,pos_type,pos_type_mask_l1,pos_type_mask_l2,pos_type_mask_l3 = gen_hier_type_2(model_args,complete_pos_tag)
      pos_tag_list.append(pos_type)
      pos_mask_l1_list.append(pos_type_mask_l1)
      pos_mask_l2_list.append(pos_type_mask_l2)
      pos_mask_list.append(pos_type_mask_l3)
      neg_tag_list.append(neg_tag)

  if model_args.type_tree.hieral_layer==2:
    return np.asarray(pos_tag_list,np.int32),np.asarray(pos_mask_l1_list,np.float32),np.asarray(pos_mask_list,np.float32),np.array(neg_tag_list,np.int32) 
  elif model_args.type_tree.hieral_layer==3:
      return np.asarray(pos_tag_list,np.int32),np.asarray(pos_mask_l1_list,np.float32),np.asarray(pos_mask_l2_list,np.float32),np.asarray(pos_mask_list,np.float32),np.array(neg_tag_list,np.int32) 
  else:
    print('single layer...')
    exit(0)
    

def gen_type_pair(model_args,data_tag_out):
  type_set = set(range(model_args.args.class_size))
  sample_size = len(data_tag_out)
  type_pair_list = []
  type_cor_score_list = []
  for i in range(sample_size):
    complete_pos_tag = list(np.nonzero(data_tag_out[i])[0])
    
    pos_tag = random.choice(complete_pos_tag)
    neg_tag = random.choice(list(type_set-set([pos_tag])))  #do not compute score with itself...
    type_pair_list.append([pos_tag,neg_tag])
    
    try:
      correlation = model_args.type_correlation[pos_tag][neg_tag]
    except:
      correlation = 0.0
    type_cor_score_list.append(correlation)
  #print type_cor_score_list[:5] 
  return np.array(type_pair_list,np.int32),np.array(type_cor_score_list,np.float32)
    