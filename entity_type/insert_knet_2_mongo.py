import sys
sys.path.append('utils')
import sys
sys.path.append("embedding")
from embedding import NMTDataRead
import numpy as np
import tensorflow as tf
import cPickle
import model_args 
from tqdm import tqdm
from pymongo import MongoClient
import random

args = model_args.args

def filter_type(type_i):
  type_list = np.nonzero(np.array(type_i))[0]
  new_type_name = set()
  old_type_name = set()
  new_type_set = set()
  for type_ in type_list:
    type_name = model_args.type_tree.id2type[type_]
    old_type_name.add(type_name)
    type_new = type_
    new_type_name.add(type_name)
    new_type_set.add(type_new)
    '''
    if (type_name in model_args.type_tree.l2_type2id or type_name in model_args.type_tree.l3_type2id) and '/other' in type_name:
      type_new = model_args.type_tree.type2id['/other']
      new_type_name.add('/other')
    else:
      type_new = type_
      new_type_name.add(type_name)
      
    new_type_set.add(type_new)'''
  
  if new_type_name!=old_type_name:
    print(new_type_name)
    print(old_type_name)
    print('---------------------------------------')
  type_ = [0]*args.class_size
  for t in new_type_set: 
    type_[t] = 1
        
  return type_
  
def main(_):
  
  train_data_reader =  NMTDataRead(args,model_args.characterEmbed,model_args.word2vecModel,args.datasets,'train')
  
  
  client = MongoClient('mongodb://192.168.3.196:27017')
  db = client['entity_typing'] # database name
  train_collection = db[args.datasets+'_train']
  ent_id = 0
  
  for items in train_data_reader.get_input_figer_chunk_train():
    
    if ent_id % 10000==0:
      print(ent_id)
      
    ent_index = items[0]
    ent_ment_surface_chars = items[1]
    ent_ment_surface_chars_lent = items[2]
    sent_wds = items[3]
    type_ = items[4]
    new_type_ = filter_type(type_)
    
    
    #we need to change the /other/animal into /other
    record = {}
    record['ent_id'] = ent_id
    record['ent_index'] = ent_index
    record['ent_ment_surface_chars'] = ent_ment_surface_chars
    record['ent_ment_surface_chars_lent'] = ent_ment_surface_chars_lent
    record['sent_wds'] = sent_wds
    record['type_'] = new_type_
    
    ent_index = map(int,ent_index)
      
    sent_pos = []
    for i in range(len(sent_wds)):
      if i < ent_index[0]:
        sent_pos.append(i-int(ent_index[0])+args.sentence_length)
      elif i>=ent_index[1] and i<ent_index[1]:
        sent_pos.append(0)
      else:
        sent_pos.append(i-int(ent_index[1])+args.sentence_length)
        
    record['sent_pos'] = sent_pos
    
    train_collection.insert_one(record)
    
    ent_id += 1
  client.close()

 
if __name__=='__main__':
  tf.app.run()