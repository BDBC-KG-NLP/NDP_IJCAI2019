# -*- coding: utf-8 -*-

import random
import numpy as np


def random_dict_keys(dict_):
  key_list = list(dict_.keys())
  random.shuffle(key_list)
  
  return key_list

def gen_hier_neg(model_args,pos_types,l1_dict,l2_dict):
  args = model_args.args
  neg_tag = set()
  
  type_set = set(range(args.class_size))
  
  
  for i_key_l1 in random_dict_keys(model_args.type_tree.l1_type2id):
    i_key_l1_id = model_args.type_tree.type2id[i_key_l1]
    if  i_key_l1_id not in l1_dict and i_key_l1_id not in pos_types:
      if i_key_l1_id not in neg_tag:
        neg_tag.add(i_key_l1_id)
     
      if len(neg_tag)==args.max_neg_type:
        return list(neg_tag)
      
  for key_l1 in random_dict_keys(l1_dict):
    for i_key_l2 in random_dict_keys(model_args.type_tree.l2_type2id):
      i_key_l2_parent = '/'+i_key_l2.split('/')[1:][0]
      
      i_key_l2_id = model_args.type_tree.type2id[i_key_l2]
      if key_l1 == i_key_l2_parent and i_key_l2_id not in pos_types:
        if i_key_l2_id not in neg_tag:
          neg_tag.add(i_key_l2_id)
        
      if len(neg_tag)==args.max_neg_type:
        return list(neg_tag)

  rand_neg = random.sample(list(type_set-neg_tag-set(pos_types)),args.max_neg_type- len(list(neg_tag)))
  neg_tag = rand_neg +  list(neg_tag)
  return neg_tag
   
def gen_hier_type(model_args,pos_types):
  new_pos_types = set()
  l1_dict= {}
  l2_dict={}
  args=model_args.args
  type_name_lists=[]
  
  for type_id in pos_types:
    type_name = model_args.type_tree.id2type[type_id]
    
    type_name_list = type_name.split('/')[1:]
    type_name_lists.append(type_name)
    if len(type_name_list)==1:
      t1 = '/'+type_name_list[0]
      
      if t1 not in l1_dict:
        l1_dict[t1] = len(l1_dict)
    else:
      t1 = '/'+type_name_list[0]
      t2 = '/'+type_name_list[0]+'/'+type_name_list[1]
      
      if t1 not in l1_dict:
        l1_dict[t1] = len(l1_dict)
      
      if t2 not in l2_dict:
        l2_dict[t2] = len(l2_dict)
      
  
  pos_type_l2= np.asarray([[args.class_size]*args.max_pos_type_l2] * model_args.args.max_pos_type_l1)
  
  pos_type_mask_l1 = np.asarray([0]*model_args.args.max_pos_type_l1)
  pos_type_mask_l2 = np.asarray([[0]*args.max_pos_type_l2]*model_args.args.max_pos_type_l1)
  
  assert(len(l1_dict)!=0)
  
  for key_l1 in l1_dict:
    l1_id = l1_dict[key_l1]
    type_id = model_args.type_tree.type2id[key_l1]
    
    pos_type_mask_l1[l1_id]=1
    
    l2_id = 0
    for key_l2 in l2_dict:
      type_id_l2 = model_args.type_tree.type2id[key_l2]
      if key_l2.startswith(key_l1):
        if type_id_l2 in pos_types:
          pos_type_l2[l1_id][l2_id] = type_id_l2
          new_pos_types.add(type_id_l2)
          pos_type_mask_l2[l1_id][l2_id]=1
          l2_id += 1
    
    #whether to add the person only?
    if model_args.args.is_add_fnode==True or l2_id ==0:
      if type_id in pos_types:
        new_pos_types.add(type_id)
        pos_type_l2[l1_id][l2_id] = type_id
        pos_type_mask_l2[l1_id][l2_id]=1
        l2_id += 1
        
   
  neg_tag = gen_hier_neg(model_args,new_pos_types,l1_dict,l2_dict)  #add as the negative ones...
  #type_set = set(range(args.class_size))
  #neg_tag = random.sample(list(type_set-set(pos_types)),args.max_neg_type) #to update the process...
  #print(neg_tag)
  return neg_tag,np.reshape(pos_type_l2,(-1,)),pos_type_mask_l1,pos_type_mask_l2