# -*- coding: utf-8 -*-

import sys
sys.path.append("evals/")
import numpy as np
import cPickle
from tqdm import tqdm
import random
from evalChunk import openSentid2aNosNoid,getaNosNo2entMenOntos
from TrieTree import TrieTree
import collections
from scipy import spatial
from sklearn import preprocessing
import argparse

def find_max_length(fname):
  sentid = 0
  temp_len = 0
  sent_lent = []
  max_length = 0
  sents = []
  with open(fname) as file_:
    for line in file_:
  
      if line in ['\n', '\r\n']:
        sent_lent.append(temp_len)
        sentid += 1
        if temp_len > max_length:
          max_length = temp_len
        temp_len = 0
        sents=[]
      else:
        sents.append(line.split(' ')[0])
        temp_len += 1
  print 'avg length:',np.average(sent_lent),'max_length:',max_length,'total sents:',sentid
  return max_length


'''
@generate the type dictionary!
'''
def typeFiger():
  type_set=set()
  
  typeId = 0
  fname = dir_path+'process/train_gold_entMen2aNosNoid.txt'
  
  for line in open(fname):
    line = line.strip()
    items = line.split('\t')
    
    for i in range(4,len(items)):
       type_i = items[i]
#       if dataset=='OntoNotes':
#         if type_i.startswith('/other'):
#           if type_i!='/other':
#             type_i = type_i.replace('/other','') 
       type_set.add(type_i)
  
  type2id = {}
  for typei in type_set:
    type2id[typei] = typeId
    typeId += 1
      
  params = {'type2id':type2id}
  id2type = {type2id[i]:i for i in type2id}
  type_weight=[]
  print 'type size:',len(type2id)
  for key in id2type:
    type_name = id2type[key]
    tt = type_name.split('/')[1:]
    print key,type_name,tt
    type_weight.append(len(tt)*1.0)
  cPickle.dump(params,open(dir_path+'type2id.p','wb'))
  cPickle.dump(type_weight,open(dir_path+'type2weight.p','wb'))

def splitData():
  fname = dir_path+'process/test_sentid2aNosNoid.txt'
  docSet=set()
  for line in open(fname):
    line = line.strip()
    items = line.split('\t')
    
    aNo = items[1].split('-')[0]
    # items[1]
    #print 'aNo:',aNo
    docSet.add(aNo)
  print 'all test doc nums:',len(docSet)
  '''
  random extract doc as train, validation, test
  '''
  nums = int(val_ratio * len(docSet))
  print nums
  validation = random.sample(list(docSet), nums)
  print validation
 
  return validation

def getSplitData_from_traning():
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path,"train")
  aNosNo2entMen = getaNosNo2entMenOntos(dir_path,"train")
  #testa = cPickle.load(open(dir_path+"figer.testa",'rb'))
  testa = splitData()
 
  testa_dict={testa[i]:i for i in range(len(testa))}
  #testb_dict={testb[i]:i for i in range(len(testb))}
  
  doc_set = set()
  
  testa_entMents=[]
  train_entMents=[]
  allEnts = 0
  testaEnts=0
  trainEnts=0
  word = []
  maxLents = 0
  sid = 0
  input_file_obj = open(dir_path+'process/'+dataset+'Data_train.txt')
  testa_outfile = open(dir_path+'features/'+dataset+'Data_testa.txt','w')
  train_outfile = open(dir_path+'features/'+dataset+'Data_train.txt','w')
  for line in tqdm(input_file_obj):
    if line in ['\n', '\r\n']:
      aNosNo = sid2aNosNo[sid]
      aNo = aNosNo.split('-')[0]   
      doc_set.add(aNo)
      entList = aNosNo2entMen[aNosNo]

      allEnts += len(entList)
      datas = '\n'.join(word) + '\n\n'
      maxLents = max(maxLents,len(word))
      if aNo in testa_dict:
        testaEnts += len(entList)
        testa_entMents.append(entList)
        testa_outfile.write(datas)
      else:
        train_entMents.append(entList)
        trainEnts += len(entList)
        train_outfile.write(datas) 
      word = []
      sid += 1
    else:
      line = line.strip()
      word.append(line)
  print 'testa ent numbers:',testaEnts
  print 'train ent numbers:',trainEnts
  print 'all test ent numbers:',allEnts
  print abs(testaEnts*1.0/allEnts - val_ratio)
  print 'document nums:',len(doc_set)
  if abs(testaEnts*1.0/allEnts) - val_ratio <= 0.01:
    cPickle.dump(testa_entMents,open(dir_path+'features/testa_entMents.p','wb'))
    cPickle.dump(train_entMents,open(dir_path+'features/train_entMents.p','wb'))
  testa_outfile.close();
  train_outfile.close()
  return testaEnts*1.0/allEnts - val_ratio


def getSplitData():
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path,"test")
  aNosNo2entMen = getaNosNo2entMenOntos(dir_path,"test")
  #testa = cPickle.load(open(dir_path+"figer.testa",'rb'))
  testa = splitData()
 
  doc_set = set()
  
  testa_entMents=[]
  testb_entMents=[]
  allEnts = 0
  testaEnts=0
  testbEnts=0
  word = []
  maxLents = 0
  sid = 0
  input_file_obj = open(dir_path+'process/'+dataset+'Data_test.txt')
  testa_outfile = open(dir_path+'features/'+dataset+'Data_testa.txt','w')
  testb_outfile = open(dir_path+'features/'+dataset+'Data_testb.txt','w')
  for line in input_file_obj:
    if line in ['\n', '\r\n']:
      aNosNo = sid2aNosNo[sid]
      #aNo = aNosNo.split('_')[0]
      aNo = aNosNo.split('-')[0]
      #print aNo
      doc_set.add(aNo)
      entList = aNosNo2entMen[aNosNo]
      #print entList
      allEnts += len(entList)
      datas = '\n'.join(word) + '\n\n'
      maxLents = max(maxLents,len(word))
      if aNo in testa:
        testaEnts += len(entList)
        testa_entMents.append(entList)
        testa_outfile.write(datas)
      else:
        testb_entMents.append(entList)
        testbEnts += len(entList)
        testb_outfile.write(datas) 
      word = []
      sid += 1
    else:
      line = line.strip()
      word.append(line)
  print 'testa ent numbers:',testaEnts
  print 'testb ent numbers:',testbEnts
  print 'all test ent numbers:',allEnts
  print abs(testaEnts*1.0/allEnts - val_ratio)
  print 'document nums:',len(doc_set)
  if abs( testaEnts*1.0/allEnts) - val_ratio <= 0.01:
    cPickle.dump(testa_entMents,open(dir_path+'features/testa_entMents.p','wb'))
    cPickle.dump(testb_entMents,open(dir_path+'features/testb_entMents.p','wb'))
  testa_outfile.close();
  testb_outfile.close()
  return testaEnts*1.0/allEnts - val_ratio


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
  return type_path_nums,flag

'''
@generate the ents
'''  
def getEnts(dir_path,tag,outtag,pronominal_words):
  print(tag,outtag)
  sid2aNosNo,aNosNo2sid = openSentid2aNosNoid(dir_path,tag)
  aNosNo2entMen = getaNosNo2entMenOntos(dir_path,tag)
  docs = set()
 # print aNosNo2entMen
  
  entMents =[]
  ent_ment = 0
  all_ent_ment = 0
  noise_ent_ments = 0
  noise_ent_ments_hier = [0]*hieral_layer
  stop_ents = 0
  for sid in tqdm(range(len(sid2aNosNo))):
    aNosNo = sid2aNosNo[sid]
    aNo = aNosNo.split('-')[0]
    docs.add(aNo)
    entList = aNosNo2entMen[aNosNo]
    new_ent_list = []
    for entItem in entList:
      #print entItem
      #entItem example: ['37', '40', [28, 126], 'Wichita , Kansas']
      
      type_path_nums,noise_flag =is_noise_data(entItem[2])
      if noise_flag:
        noise_ent_ments+= 1
      for i in range(hieral_layer):
        if type_path_nums[i] >=2:
          noise_ent_ments_hier[i] += 1
                              
      if entItem[3].lower() not in pronominal_words:#and noise_flag ==False: 
        new_ent_list.append(entItem)
      else:
        stop_ents += 1
        #print entItem
    entMents.append(list(new_ent_list))
    ent_ment += len(new_ent_list)
    all_ent_ment += len(entList)
  print 'tag:',tag
  print 'sentence num:',len(sid2aNosNo)
  print 'stop ents num:',stop_ents
  print 'all entMents:',all_ent_ment
  print 'noise ent mentions:',noise_ent_ments, 'noise 占总数的比例:', noise_ent_ments*1.0/all_ent_ment
  print 'noise_ent_ments_hier:',noise_ent_ments_hier,np.asarray(noise_ent_ments_hier)/(noise_ent_ments*1.0) 
  print'all doc nums:',len(docs) 
  print 'traing ent nums:',ent_ment
  print '--------------------------------'
  print('save datasets..')
  cPickle.dump(entMents,open(dir_path+'features/'+outtag+'_entMents.p','wb'))


def get_ent_mention_nums(tag,lent_list):
  entMents = cPickle.load(open(dir_path+'features/'+tag+'_entMents.p','rb'))
  print 'tag:',tag
  print 'sent nums:',len(entMents)
    
  all_ents = 0.0
  ent_lent_max = 0
  max_lent_ent_name = ''
  noise_ent_ments = 0
  
  for entList in tqdm(entMents):
    all_ents += len(entList)
    
    for ent in entList:
      #['5', '7', [10, 3], 'Maggie Steber']
      type_path_nums,noise_flag =is_noise_data(ent[2])
      if noise_flag:
        noise_ent_ments+= 1
        
      ent_name = ent[3]
      lent_list.append(len(ent_name))
      if len(ent_name) > ent_lent_max:
        ent_lent_max = len(ent_name)
        max_lent_ent_name = ent_name
  
  #print len(entMents)
  print 'ent nums:',all_ents,'noise ent:',noise_ent_ments*1.0/all_ents
  print ent_lent_max,max_lent_ent_name
  return  all_ents,lent_list


def get_type_correlation(tag):
  print 'load '+tag+' data set ...'
  entMents = cPickle.load(open(dir_path+'features/'+tag+'_entMents.p','rb'))
  type2ent = {}
  rel_type2ent = {}
  rel_id2type = {}
  rel_type2id = {}
  rel_id = 0
  
  for entList in tqdm(entMents):
    for ent in entList:
      #['5', '7', [10, 3], 'Maggie Steber']
      type_list = ent[2]
      for typei in type_list:
        rel_type_name = id2type[typei].lower()
        #if True:
        if rel_type_name.count('/') == 2:
          if rel_type_name not in rel_type2id:
            rel_id2type[rel_id] = rel_type_name
            rel_type2id[rel_type_name] = rel_id
            rel_id += 1
            
          t_id = rel_type2id[rel_type_name]
          if t_id not in rel_type2ent:
            rel_type2ent[t_id] = collections.defaultdict(int)
            
          rel_type2ent[t_id][ent[-1]] += 1
                
        reduce_type = id2type[typei].lower()
        if '/location' not in reduce_type and '/organization' not in reduce_type and '/person' not in reduce_type:
          reduce_type='/other'
        else:
          reduce_type = '/'+reduce_type.split('/')[1]
          
        if reduce_type not in type2ent:
          type2ent[reduce_type] = collections.defaultdict(int)
        
        type2ent[reduce_type][ent[-1]] += 1
  
  top_type_lent  = len(rel_type2id)
  print top_type_lent
  
  rel_scores = [[0]*top_type_lent for i in range(top_type_lent)]
  for i in tqdm(range(top_type_lent)):
    ti_ents = set(rel_type2ent[i].keys())
    ti_name = rel_id2type[i]
    for j in range(len(rel_type2ent)):
      tj_name = rel_id2type[j]
      if ti_name not in tj_name and tj_name not in ti_name:
        tj_ents = set(rel_type2ent[j].keys())
        ti_tj_ents = ti_ents & tj_ents
        
        rel_scores[i][j] = (len(ti_tj_ents)*1.0/len(ti_ents)+len(ti_tj_ents)*1.0/len(tj_ents))/2
        rel_scores[j][i] = rel_scores[i][j]
      
#  print np.sum(rel_scores,-1)
#  print np.average(np.sum(rel_scores,-1))
#  print np.average(rel_scores)
  
  all_ents = 0.0
  for rid in rel_type2ent:
    all_ents += np.sum(rel_type2ent[rid].values())
    
  new_rel_score = []
  for rid in rel_type2ent:
    correlated_type_num = np.size(np.nonzero(rel_scores[rid]))
    if correlated_type_num == 0:
      new_rel_score.append(0)
    else:
      new_rel_score.append(np.sum(rel_scores,-1)[rid]/correlated_type_num)
  
  
  
  weight_rel_score=[]
  for rid in rel_id2type:
    type_ent_nums = np.sum(rel_type2ent[rid].values())
    weight_rel_score.append(type_ent_nums/all_ents * new_rel_score[rid])
    print rid, rel_id2type[rid],len(rel_type2ent[rid]),type_ent_nums, type_ent_nums/all_ents,new_rel_score[rid]
  
  print '------------'
  print np.sum(weight_rel_score)
  exit(0)
  
  for key in type2ent:
    print key,len(type2ent[key])
  type_nums = len(type2ent)
  reduce_type2id = {'/person':0,'/location':1,'/organization':2,'/other':3} #都是按照这个来进行排列的呢！
  reduce_id2type = {reduce_type2id[typei]:typei for typei in reduce_type2id}
  print reduce_id2type
  
  score = [[0]*type_nums for i in range(type_nums)]
  
  for i in range(type_nums):
    print i
    typei = reduce_id2type[i]
    ti_ents = set(type2ent[typei].keys())
    
    for j in range(i+1,type_nums):
      typej = reduce_id2type[j]
      tj_ents = set(type2ent[typej].keys())
      
      ti_tj_ents = ti_ents & tj_ents
      
      score[i][j] = (len(ti_tj_ents)*1.0/len(ti_ents)+len(ti_tj_ents)*1.0/len(tj_ents))/2
      score[j][i] = score[i][j]
  print score
  print reduce_type2id
  return reduce_type2id,score
        
        
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, help='[Wiki, OntoNotes, BBN]', required=True)
  parser.add_argument('--val_ratio', type=float, help='0.01 for Wiki and 0.05 for OntoNotes and BBN', required=True)
  args = parser.parse_args()
  
  pronominal_words={}
#  for line in open('data/pronominal_words.txt'):
#    line = line.strip()
#    pronominal_words[line.lower()]=1
  
  root_path = 'data/KDD/'
  dataset = args.dataset
  val_ratio = args.val_ratio
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
  
  dir_path = root_path + dataset+'/'
  #from loadWordEmbed import wordEmbedding
  
  tag ='test'
  t = cPickle.load(open(dir_path+'type2id.p'))['type2id']
  t = cPickle.load(open(dir_path+'type2id.p'))['type2id']
  id2type = {t[typei]:typei for typei in t}
  print id2type
  getEnts(dir_path,'test','test',pronominal_words)
  
  '''
  tag = 'train'
  
  typeFiger()
  
  t = cPickle.load(open(dir_path+'type2id.p'))['type2id']
  id2type = {t[typei]:typei for typei in t}
  print id2type
    
  getEnts(dir_path,'train','train',pronominal_words)
  #getEnts(dir_path,'test','testb',pronominal_words) #==> we treat all the dataset as the 
  
  
  tag = True
  while tag:    
    print '----------------------'
    sim = getSplitData()
    if abs(sim) <=0.001:
      tag=False
    print '----------------------'
  '''
  
  '''
  data_fname = dir_path+'features/'+dataset+'Data_train.txt'
  sid = 0
  with open(data_fname) as file_:
    for line in file_:
      if line in ['\n', '\r\n']:
        sid += 1
  print 'train:',sid
  
  
  data_fname = dir_path+'features/'+dataset+'Data_testa.txt'
  sid = 0
  with open(data_fname) as file_:
    for line in file_:
      if line in ['\n', '\r\n']:
        sid += 1
  print 'testa:',sid
  
  data_fname = dir_path+'features/'+dataset+'Data_testb.txt'
  sid = 0
  with open(data_fname) as file_:
    for line in file_:
      if line in ['\n', '\r\n']:
        sid += 1
  print sid
  '''
  '''
  t = cPickle.load(open(dir_path+'type2id.p'))['type2id']
  id2type = {t[typei]:typei for typei in t}
  getEnts(dir_path,'train','train',pronominal_words)
  getEnts(dir_path,'test','test',pronominal_words)
  '''
  '''
  lent_list = []
  t = cPickle.load(open(dir_path+'type2id.p'))['type2id']
  id2type = {t[typei]:typei for typei in t}
  
  testa_all_ents,lent_list = get_ent_mention_nums('testa',lent_list)
  testb_all_ents,lent_list = get_ent_mention_nums('testb',lent_list)
  train_all_ents,lent_list = get_ent_mention_nums('train',lent_list)  
  #print np.max(lent_list),np.average(lent_list)
  #print testa_all_ents,testb_all_ents,train_all_ents
  '''
  '''
  fname = dir_path + 'features/'+dataset+'Data_testa.txt'
  find_max_length(fname)
  
  fname = dir_path + 'features/'+dataset+'Data_testb.txt'
  find_max_length(fname)
  
  fname = dir_path + 'features/'+dataset+'Data_train.txt'
  find_max_length(fname)
  '''
  '''
  t = cPickle.load(open(dir_path+'type2id.p'))['type2id']
  id2type = {t[typei]:typei for typei in t}
  reduce_type2id,score = get_type_correlation('train') '''
  #ret_param={'type2id':reduce_type2id,'score':score}
  #cPickle.dump(ret_param,open(dir_path+"type_correlation.p",'wb'))
  