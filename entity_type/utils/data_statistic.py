# -*- coding: utf-8 -*-
import sys
import cPickle
from tqdm import tqdm
import collections
from loadWordEmbed import wordEmbedding
import tensorflow as tf

def is_noise_data(type_list):
  type_set = set()
  type_path_nums = [0]*hieral_layer
  type_path_name = []
  flag = False
  
  for typei in type_list:
    type_name = id2type[typei]
    type_path = type_name.split('/')[1:]
    type_path_nums[len(type_path)-1]+=1
    type_path_name.append(type_name)
    
    if type_path_nums[len(type_path)-1] >=2:
      flag =  True
    type_set.add(type_name)
  #if flag:
  #  print entName,type_path_name
  return type_path_name,type_path_nums,flag


def get_max_type_list(entMents):
  max_type_lent = 0
  for sid in range(len(entMents)):
    
    entList = entMents[sid]
    #ent_no,temp_ent_mention_mask,temp_type = self .getFigerEntTags(entList,allid%batch_size,ent_no)
    for i in range(len(entList)):
      ent = entList[i]
#      ent_start= int(ent[0])
#      ent_end = int(ent[1])
      typeList = ent[2]
      
      max_type_lent = max(len(typeList),max_type_lent)
  print max_type_lent

def  get_type_correlation(entMents):
  type2ent = {}
 
  for entList in tqdm(entMents):
    for ent in entList:
      #['5', '7', [10, 3], 'Maggie Steber']
      type_list = ent[2]
      
      for type_i in type_list:
        if type_i not in type2ent:
          type2ent[type_i] = collections.defaultdict(int)
        
        type2ent[type_i][ent[-1]] += 1
  
  type_nums = len(type2id)             
  score = [[0]*type_nums for i in range(type_nums)]
  
  for i in tqdm(range(type_nums)):
    ti_ents = set(type2ent[i].keys())
     
    for j in range(i+1,type_nums):
      tj_ents = set(type2ent[j].keys())
      
      ti_tj_ents = ti_ents & tj_ents
      
      score[i][j] = (len(ti_tj_ents)*1.0/len(ti_ents)+len(ti_tj_ents)*1.0/len(tj_ents))/2
      score[j][i] = score[i][j]
   
  type2num = { key:len(type2ent[key]) for key in type2ent}
  params = {'score':score,'type2num':type2num}
  
  cPickle.dump(params,open(dir_path+'total_type_correlation.p','wb'))
  
def get_ent_surface(entMents):
  ent_dict ={}
  for line in tqdm(entMents):
    for ent_item in line:
      ent_surface_name = ent_item[-1].lower()
      
      if ent_surface_name not in ent_dict:
        ent_dict[ent_surface_name] = 1
  return ent_dict

def get_ent_in_dict(entMents):
  print dataset
  flags = tf.app.flags
  flags.DEFINE_string("dir_path",'data/KDD/',"data path")
  flags.DEFINE_string("datasets",dataset,"data")
  
  args = flags.FLAGS
  
  wordEmbed = wordEmbedding('train',args)
  
  for line in tqdm(entMents):
    for ent_item in line:
      ent_surface_name = ent_item[-1].split(' ')
      flag = True
      for wd in ent_surface_name:
        if wd in wordEmbed.vocab or wd.lower() in wordEmbed.vocab:
          continue
        else:
          flag = False
          break
      if flag==False:
        print 'wrong ent:',ent_surface_name
      
      
      
#  for wd in word_pairs:
#    flag,ids = wordEmbed.get_vocab_id(wd)
#    if wd not in wd_2_id:
#      wd_2_id[wd] = ids
  
if __name__ =="__main__":
  dataset = sys.argv[1]
  
  dir_path = 'data/KDD/'+dataset+'/'
  
  test_entMents   = cPickle.load(open(dir_path+'features/testb_entMents.p','rb')) 
  test_ent_dict = get_ent_surface(test_entMents)
  
  get_max_type_list(test_entMents)
  
  #get_ent_in_dict(test_entMents)
  
  train_entMents = cPickle.load(open(dir_path+'features/'+'train_entMents.p','rb'))
  train_ent_dict = get_ent_surface(train_entMents)
  
  get_max_type_list(train_entMents)
  
#  
#  test_in_train = 0.0
#  for ent in test_ent_dict:
#    if ent in train_ent_dict:
#      test_in_train += 1
#  print test_in_train, len(test_entMents), test_in_train/len(test_entMents)
  
  '''
  type_nums = 0
  if dataset=='Wiki':
    type_nums = 113
    hieral_layer=2
  elif dataset == 'OntoNotes':
    type_nums = 89
    hieral_layer=3
  elif dataset =='BBN':
    type_nums = 47
    hieral_layer=2

  type2id = cPickle.load(open(dir_path+'type2id.p'))['type2id']
  id2type ={type2id[val]:val for val in type2id}
  '''
  '''
  test_ents = 0.0
  test_wrong = 0.0
  for line in test_entMents:
    for ent_item in line:
      type_list = map(int,ent_item[2])
      type_path_name,type_path_nums,flag = is_noise_data(type_list)
      test_ents += 1
      if flag==False:
        test_wrong += 1
        print ent_item
        print type_list
        print type_path_name,type_path_nums,flag
        print '----------------'
  print test_wrong, test_ents, test_wrong/test_ents    
  '''
  #get_type_correlation(train_entMents) 
  