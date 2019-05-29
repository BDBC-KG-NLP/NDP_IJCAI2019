# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:53:21 2018

@author: wujs
"""
from evals import MentTrieTree
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import model_args

args = model_args.args

def get_ret_type_error(gold_tag):
  type_path_nums=[]
  for typei in gold_tag:
    type_name = args.type_tree.id2type[typei]
    type_path = type_name.split('/')[1:]
    print(type_name,type_path)
    type_path_nums[len(type_path)-1]+=1
      
  return type_path_nums

def get_level_error(entMents):
  all_ents = 0.0
  level_error = args.type_tree.hieral_layer*[0]
  for line in entMents:
    for ent_item in line:
      all_ents+= 1
      
      type_list = ent_item[2]
      
      type_path_nums=get_ret_type_error(type_list)
      for leveli in range(type_path_nums):
        i_num = type_path_nums[leveli]
        if i_num >=1:
          level_error[leveli] += 1
      
  print(np.asarray(level_error)/all_ents)


def get_leaf_node(type_list):
  '''
  @we only reserve the leaf types
  '''
  t=[0,0,0]
  
  tree = MentTrieTree()
  type_name_list= []
  for  ti in type_list:
    tname = model_args.type_tree.id2type[ti]
    tree.add(tname)
    type_name_list.append(tname)
    
  for tname in type_name_list:
    if tree.is_leaf(tname):
      
      if tname in model_args.type_tree.l1_type2id:
        t[0]=1
      elif tname in model_args.type_tree.l2_type2id:
        t[1]=1
      else:
        t[2]=1
  return t
        

def main(_):
  
  print(args.datasets)
  testb_data_reader = model_args.testb_data_reader
  testa_data_reader = model_args.testa_data_reader
  train_collection = model_args.train_collection
  

  
  single_type_ents=0
  level2_type_ents=0
  level3_type_ents=0
  for line in tqdm(testa_data_reader.entMents):
    for ent_item in line:
      type_list = ent_item[2]
      
      type_level_i = get_leaf_node(type_list)
      single_type_ents += type_level_i[0]
      level2_type_ents += type_level_i[1]
      level3_type_ents += type_level_i[2]      

  print('testa:',single_type_ents,level2_type_ents,level3_type_ents)
  print('..................................') 
  
  
  single_type_ents=0
  level2_type_ents=0
  level3_type_ents=0
  for line in tqdm(testb_data_reader.entMents):
    for ent_item in line:
      type_list = ent_item[2]
      
      type_level_i = get_leaf_node(type_list)
      single_type_ents += type_level_i[0]
      level2_type_ents += type_level_i[1]
      level3_type_ents += type_level_i[2]      
  
  print('testb:',single_type_ents,level2_type_ents,level3_type_ents)
  print('..................................') 
  
  single_type_ents=0
  level2_type_ents=0
  level3_type_ents=0
  for record in tqdm(train_collection.find({})):
    type_list = np.nonzero(record['type_'])[0]
    
    type_level_i = get_leaf_node(type_list)
    single_type_ents += type_level_i[0]
    level2_type_ents += type_level_i[1]
    level3_type_ents += type_level_i[2]
      
    
  print('train:',single_type_ents,level2_type_ents,level3_type_ents)
  print('..................................') 
  

 
if __name__=='__main__':
  tf.app.run()