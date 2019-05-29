# -*- coding: utf-8 -*-

import random
import numpy as np

def random_dict_keys(dict_):
  key_list = list(dict_.keys())
  random.shuffle(key_list)
  
  return key_list


def gen_hier_neg_2(model_args,pos_types,l1_dict,l2_dict,l3_dict):
  neg_tag = []
  neg_tag_name = []
  type_set = set(range(model_args.args.class_size))
  pos_type_name = []
  for ti in pos_types:
    pos_type_name.append(model_args.type_tree.id2type[ti])
  
  for i_key_l1 in random_dict_keys(model_args.type_tree.l1_type2id):
    i_key_l1_id = model_args.type_tree.type2id[i_key_l1]
    if i_key_l1_id not in pos_types and i_key_l1_id not in l1_dict:
      if i_key_l1_id not in neg_tag:
        neg_tag.append(i_key_l1_id)
        neg_tag_name.append(['l1:',i_key_l1])
        
      if len(neg_tag)==model_args.args.max_neg_type:
        return neg_tag
#  print(neg_tag)
#  print(len(model_args.type_tree.l1_type2id))
#  print(len(pos_type_name))
#  print(pos_type_name)
#  print('-------------------------------')
  for key_l1 in random_dict_keys(l1_dict):
    for i_key_l2 in random_dict_keys(model_args.type_tree.l2_type2id):
      i_key_l2_parent = '/'+i_key_l2.split('/')[1:][0]
      i_key_l2_id = model_args.type_tree.type2id[i_key_l2]
      
      if key_l1 == i_key_l2_parent and i_key_l2_id not in pos_types and i_key_l1_id not in l2_dict: #add 2-th layer nearest brothers
        if i_key_l2_id not in neg_tag:
          neg_tag.append(i_key_l2_id)
          neg_tag_name.append(['l2:',i_key_l2])     
      if len(neg_tag)==model_args.args.max_neg_type:
        return neg_tag
         
  for key_l2 in random_dict_keys(l2_dict):
    for i_key_l3 in random_dict_keys(model_args.type_tree.l3_type2id):
      i_key_l3_parent = '/'+'/'.join(i_key_l3.split('/')[1:][0:-1])
      i_key_l3_id = model_args.type_tree.type2id[i_key_l3]
      
      if key_l2 == i_key_l3_parent and i_key_l3_id not in pos_types: # add 3-th layer nearest brothers
        if i_key_l3_id not in neg_tag:
          neg_tag.append(i_key_l3_id)
          neg_tag_name.append(['l3:',i_key_l3])
       
      if len(neg_tag)==model_args.args.max_neg_type:
        return neg_tag
  
  
  
  rand_neg = random.sample(list(type_set-set(neg_tag)-set(pos_types)),model_args.args.max_neg_type-len(neg_tag))
  neg_tag = rand_neg +  neg_tag
  return neg_tag 
  
def gen_hier_type_2(model_args,pos_types):
  new_pos_types = []
  args = model_args.args
  l1_dict={}
  l2_dict={}
  l3_dict={}
  type_set = set(range(model_args.args.class_size))
  
  for type_id in pos_types:
    type_name = model_args.type_tree.id2type[type_id]
    
    type_name_list = type_name.split('/')[1:]
    
    if len(type_name_list)==1:
      t1 = '/'+type_name_list[0]
      
      if t1 not in l1_dict:
        l1_dict[t1] = len(l1_dict)
    elif len(type_name_list)==2:
      t1 = '/'+type_name_list[0]
      t2 = '/'+type_name_list[0]+'/'+type_name_list[1]
      
      if t1 not in l1_dict:
        l1_dict[t1] = len(l1_dict)
      
      if t2 not in l2_dict:
        l2_dict[t2] = len(l2_dict)
    else:
      t1 = '/'+type_name_list[0]
      t2 = '/'+type_name_list[0]+'/'+type_name_list[1]
      t3 = '/'+type_name_list[0]+'/'+type_name_list[1]+'/'+type_name_list[2]
      
      if t1 not in l1_dict:
        l1_dict[t1] = len(l1_dict)
      
      if t2 not in l2_dict:
        l2_dict[t2] = len(l2_dict)
        
      if t3 not in l3_dict:
        l3_dict[t3] = len(l3_dict)
  
  pos_type_mask_l1 = np.zeros((args.max_pos_type_l1))
  pos_type_mask_l2 = np.zeros((args.max_pos_type_l1,args.max_pos_type_l2))
  pos_type_mask_l3 =  np.zeros((args.max_pos_type_l1,args.max_pos_type_l2,args.max_pos_type_l3))
  
  pos_type_l3= np.ones((args.max_pos_type_l1,args.max_pos_type_l2,args.max_pos_type_l3))*args.class_size
  
  for key_l1 in l1_dict:
    l1_id = l1_dict[key_l1]
    type_id_l1 = model_args.type_tree.type2id[key_l1]
    
    pos_type_mask_l1[l1_id]=1
    
    l2_id = 0
    for key_l2 in l2_dict:
      type_id_l2 = model_args.type_tree.type2id[key_l2]
      if key_l2.startswith(key_l1):
        l3_id = 0
        pos_type_mask_l2[l1_id][l2_id]=1

        for key_l3 in l3_dict:
          type_id_l3 = model_args.type_tree.type2id[key_l3]
          if key_l3.startswith(key_l2):
            pos_type_l3[l1_id][l2_id][l3_id]=type_id_l3
            new_pos_types.append(type_id_l3)
            pos_type_mask_l3[l1_id][l2_id][l3_id]=1
            l3_id += 1
        
        if model_args.args.is_add_fnode==True or l3_id==0 or '/other' in key_l1:
          pos_type_l3[l1_id][l2_id][l3_id]=type_id_l2
          new_pos_types.append(type_id_l2)
          #print('type_id_l2:',type_id_l2)
          pos_type_mask_l3[l1_id][l2_id][l3_id]=1
          l3_id += 1
          
        l2_id += 1
        
    if model_args.args.is_add_fnode==True or l2_id==0 or '/other' in key_l1:
      pos_type_mask_l2[l1_id][l2_id]=1
      pos_type_l3[l1_id][l2_id][0] = type_id_l1
      new_pos_types.append(type_id_l1)
      #print('type_id_l1:',type_id_l1)
      pos_type_mask_l3[l1_id][l2_id][0]=1
      l2_id += 1
  neg_tag = random.sample(list(type_set-set(new_pos_types)),model_args.args.max_neg_type)
  #neg_tag =gen_hier_neg_2(model_args,new_pos_types,l1_dict,l2_dict,l3_dict)
  return neg_tag,np.reshape(pos_type_l3,(-1,)),pos_type_mask_l1,pos_type_mask_l2,pos_type_mask_l3


  