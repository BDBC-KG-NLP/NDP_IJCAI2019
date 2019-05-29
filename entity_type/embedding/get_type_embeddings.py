#from __future__ import print_function
# -*- coding: utf-8 -*-
'''
@editor: wujs
function: chunk
revise: 2017/4/17
'''
import sys
sys.path.append("evals/")
import sys
import numpy as np
import cPickle
import gzip
from tqdm import tqdm
import random

class NMTDataRead(object):
  def __init__(self,args,characterEmbed,word2vecModel,dataset,tag):
    self.args = args
    self.max_sentence_length = self.args.sentence_length
    self.word2vecModel = word2vecModel
    self.characterEmbed = characterEmbed
    self.dataset = dataset
    print(dataset,tag)
    
    self.tag  =tag
  
    if dataset in ['OntoNotes','BBN','Wiki']:
      self.dir_path = 'data/KDD/'+dataset+'/'
      self.data_fname = self.dir_path+'features/'+dataset+'Data_'+tag+'.txt'
      self.entMents = cPickle.load(open(self.dir_path+'features/'+tag+'_entMents.p','rb'))
      self.ent_ment_nums = self.get_ent_ment_nums()
      
    if self.args.is_test==True:
      self.entMents = cPickle.load(open(self.dir_path+'features/'+tag+'_entMents.p','rb'))
      self.ent_ment_nums = self.get_ent_ment_nums()
          
    
    if dataset in ['Wiki']:
      self.dir_path = 'data/KDD/'+dataset+'/'
      
      if tag =='testa':
        self.data_fname = self.dir_path+'features/'+dataset+'Data_'+'valid'+'.txt'
        self.entMents = cPickle.load(open(self.dir_path+'features/'+'valid'+'_entMents.p','rb'))
        self.ent_ment_nums = self.get_ent_ment_nums()
        print('ent_ment_nums:',self.ent_ment_nums,' entMents:',len(self.entMents))
        print('--------------------------')
        
      elif tag =='testb':
        self.data_fname = self.dir_path+'features/'+dataset+'Data_'+'test'+'.txt'
        self.entMents = cPickle.load(open(self.dir_path+'features/'+'test'+'_entMents.p','rb'))
        self.ent_ment_nums = self.get_ent_ment_nums()
        print('ent_ment_nums:',self.ent_ment_nums,' entMents:',len(self.entMents))
        print('--------------------------')
    
  def get_ent_ment_nums(self):
    ent_ment_nums = 0 
    for line in self.entMents:
      ent_ment_nums += len(line)
    return ent_ment_nums
      
      
  def openFile(self,fileName):
    if fileName.endswith('gz'):
      return gzip.open(fileName,'r')
    else:
      return open(fileName)
  
  def find_max_length(self,fname):
    sentid = 0
    temp_len = 0
    max_length = 0
    max_sent= []
    sents = []
    with open(fname) as file_:
      for line in file_:
        if line in ['\n', '\r\n']:
          sentid += 1
          if temp_len > max_length:
            max_length = temp_len
            max_sent = sents
          temp_len = 0
          sents=[]
        else:
          sents.append(line.split(' ')[0])
          temp_len += 1
    print('max_length:',max_length)
    print(max_sent)
    print('total sents:',sentid)
    return max_length


  def pos(self,tag): 
    one_hot = np.zeros(5)
    if tag == 'NN' or tag == 'NNS':
        one_hot[0] = 1
    elif tag == 'FW':
        one_hot[1] = 1
    elif tag == 'NNP' or tag == 'NNPS':
        one_hot[2] = 1
    elif 'VB' in tag:
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot


  #our goal is the predict chunk
  def chunk(self,tag):
    one_hot = np.zeros(5)
    if 'NP' in tag:
        one_hot[0] = 1
    elif 'VP' in tag:
        one_hot[1] = 1
    elif 'PP' in tag:
        one_hot[2] = 1
    elif tag == 'O':
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot

  def getFigerEntTags(self,entList,sid,ent_no):
    ent_mention_mask=[]
    ent_type = []
    for i in range(len(entList)):
      ent = entList[i]
      ent_start= int(ent[0])
      ent_end = int(ent[1])
      typeList = ent[2]
      
      
      ent_mention_mask.append([sid,ent_start,ent_end])
      
      type_ = [0]*self.args.class_size
      for t in typeList: 
        type_[t] = 1
      #print type_
      ent_type.append(list(type_))
      ent_no += 1
    assert(len(ent_mention_mask)==len(ent_type))
    return ent_no,ent_mention_mask,ent_type

  def get_input_figerTest_chunk(self,batch_size):
    allid=0  #sentence nums in all iterations.
    words = [] #words in a sentence
    
    type_batch =[]    #entity type tag batch
    sentence_batch=[] #sentence batch
    ent_mention_index_batch=[] #entity mention index in sentence batch
    ent_mention_surface_name_batch=[] #entity mention surface name
    ent_mention_surface_name_lent_batch=[]
    retParams= []  #all test batches

    
    ent_no=0
    sentence_length = 0
    flag = True
    sid = -1
    with open(self.data_fname) as file_:
      for line in tqdm(file_):
        if flag == False:
          break
        if line in ['\n', '\r\n']:
          sid += 1
          #if self.word_type =='Id':
          if sentence_length < self.max_sentence_length:
            words += [0]*(self.max_sentence_length - sentence_length)
          elif sentence_length > self.max_sentence_length:
            print(sentence_length)
          #print words 
          assert(len(words)==self.max_sentence_length)
          
          entList = self.entMents[sid]
          allid += 1
          '''
          @2017/11/6 我们需要去产生一样大小的batch size 去训练entity loss
          @对于每一个ent 我们都给出相应的sentence，方便之后进行mention 和 ctx 进行查找
          '''
          for i in range(len(entList)):
            ent = entList[i]
            ent_start= int(ent[0])
            ent_end = int(ent[1])
            typeList = ent[2]
            
            #we need to add the surface name, to generate the entity embeddings
            ent_mention_index_batch.append([ent_no%batch_size,ent_start,ent_end])
            ent_ment_surface_chars=[]
            ent_ment_surface_chars_lent =self.args.characterLent 
            for char in ent[3]:
              ent_ment_surface_chars.append(self.characterEmbed[char])
            
            lent_chars = len(ent_ment_surface_chars)
            
            if lent_chars < ent_ment_surface_chars_lent:
              ent_ment_surface_chars += [0]*(ent_ment_surface_chars_lent-lent_chars)
            elif lent_chars > ent_ment_surface_chars_lent:
              ent_ment_surface_chars = ent_ment_surface_chars[0:ent_ment_surface_chars_lent]
              lent_chars = ent_ment_surface_chars_lent
            
            ent_mention_surface_name_batch.append(ent_ment_surface_chars)
            ent_mention_surface_name_lent_batch.append(lent_chars)
            
            type_ = [0]*self.args.class_size
            for t in typeList: 
              type_[int(t)] = 1
            assert(np.sum(type_)!=0)
            #print type_
            type_batch.append(list(type_))
            ent_no += 1
            #print 'ent_no:',ent_no, ' self.sentenceNums:',self.sentenceNums,' sid:',sid
            sentence_batch.append(words)
            
            #print 'ent_mention_surface_name_batch:',np.asarray(ent_mention_surface_name_batch,dtype=np.int32)
            if ent_no%batch_size==0 or (ent_no==self.ent_ment_nums):
              if len(sentence_batch)!=0:
                retParams.append([ent_mention_index_batch,\
                                  np.asarray(ent_mention_surface_name_batch,dtype=np.int32),\
                                  np.asarray(ent_mention_surface_name_lent_batch,dtype=np.int32),\
                                  np.asarray(sentence_batch),\
                                  np.asarray(type_batch, dtype=np.float32)])
  
              sentence_batch=[]
              type_batch=[]
              ent_mention_index_batch=[]
              ent_mention_surface_name_batch=[]
              ent_mention_surface_name_lent_batch=[]
              
          sentence_length = 0  #we need to start a new setence
          words = []
        else:
          wd = line.split()[0]
          is_vocab,temp = self.word2vecModel.get_vocab_id(wd)
          if is_vocab == True:
            sentence_length += 1
            words.append(temp)
    return retParams
  
  def get_input_figer_chunk_train(self):
    words = []

    ent_no=0
    sentence_length = 0
    sid = -1
    flag = False
    with open(self.data_fname) as file_:
      for line in tqdm(file_):
        
        if line in ['\n', '\r\n']:
          sid += 1
          entList = self.entMents[sid]
          #ent_no,temp_ent_mention_mask,temp_type = self.getFigerEntTags(entList,allid%batch_size,ent_no)
          for i in range(len(entList)):
            ent = entList[i]
            ent_start= int(ent[0])
            ent_end = int(ent[1])
            typeList = map(int,ent[2])

            ent_ment_surface_chars=[]
            
            ent_ment_surface_chars_lent =self.args.characterLent 
            
            for char in ent[3]:
              ent_ment_surface_chars.append(self.characterEmbed[char])
              
            lent_chars = len(ent_ment_surface_chars)
            
            if lent_chars < ent_ment_surface_chars_lent:
              ent_ment_surface_chars += [0]*(ent_ment_surface_chars_lent-lent_chars)
            elif lent_chars > ent_ment_surface_chars_lent:
              lent_chars=ent_ment_surface_chars_lent
              ent_ment_surface_chars = ent_ment_surface_chars[:ent_ment_surface_chars_lent]

            
            type_ = [0]*self.args.class_size
            for t in typeList: 
              type_[t] = 1
            assert(np.sum(type_)!=0)
            yield [[ent_start,ent_end], ent_ment_surface_chars,lent_chars,words,type_]
            
            if ent_no==self.ent_ment_nums:
              flag = True
              break
            
          if flag == True:
            print('load training finish:', self.ent_ment_nums)
            break
          sentence_length = 0
          words = []
        else:
          #assert (len(line.split()) == 3)  #only has Word,pos_tag
          wd = line.split()[0]
          is_vocab,temp = self.word2vecModel.get_vocab_id(wd)
          if is_vocab == True:
            sentence_length += 1
            words.append(temp)
  
  def get_shuffle_train_data(self,batch_size,training_data_nums,collection_data):
    #ent_id_list = range(training_data_nums)
    #random.shuffle(ent_id_list)
    
    random_type_batch =[]
    random_ent_mention_index_batch =[]
    ent_mention_surface_name_batch=[]
    ent_mention_surface_name_lent_batch=[]
    sentence_batch=[]
    random_ent_id = []
    ent_no=-1 

    random.shuffle(collection_data)
      
    for record in collection_data:
    #for record in collection.find({}):
      if record !=None:
        ent_id = record['ent_id']
        ent_mention_index = record['ent_index']
        type_ = map(int,record['type_'])
        
        assert(np.sum(type_)!=0)
        ent_no  += 1
        random_ent_mention_index_batch.append([ent_no%batch_size,ent_mention_index[0],ent_mention_index[1]])  
        ent_mention_surface_name_batch.append(record['ent_ment_surface_chars'])
        sentence_batch.append(record['sent_wds'])
        ent_mention_surface_name_lent_batch.append(record['ent_ment_surface_chars_lent'])
        random_type_batch.append(type_)  
        random_ent_id.append(ent_id)  
        
        flag = ((ent_no+1) %batch_size==0 or (ent_no+1)== training_data_nums)
        
        
        if flag:
          if len(random_ent_mention_index_batch)!=0:
            assert(len(random_ent_mention_index_batch)==len(sentence_batch))
            
            yield random_ent_mention_index_batch,\
                    np.asarray(ent_mention_surface_name_batch,dtype=np.int32),\
                    np.asarray(ent_mention_surface_name_lent_batch,dtype=np.int32),\
                    np.asarray(sentence_batch),\
                    np.asarray(random_type_batch,np.float32),\
                    np.asarray(random_ent_id,np.int32)
          
          random_type_batch =[]
          random_ent_mention_index_batch =[]
          ent_mention_surface_name_batch=[]
          ent_mention_surface_name_lent_batch=[]
          sentence_batch=[]
          random_ent_id = []
          
  def get_shuffle_train_data_Peng(self,batch_size,training_data_nums,collection_data):
    #ent_id_list = range(training_data_nums)
    #random.shuffle(ent_id_list)
    
    random_type_batch =[]
    random_ent_mention_index_batch =[]
    ent_mention_surface_name_batch=[]
    ent_mention_surface_name_lent_batch=[]
    sentence_batch=[]
    pos_batch = []
    random_ent_id = []
    ent_no=-1 

    random.shuffle(collection_data)
      
    for record in collection_data:
    #for record in collection.find({}):
      if record !=None:
        ent_id = record['ent_id']
        ent_mention_index = record['ent_index']
        type_ = map(int,record['type_'])
        
        assert(np.sum(type_)!=0)
        ent_no  += 1
        random_ent_mention_index_batch.append([ent_no%batch_size,ent_mention_index[0],ent_mention_index[1]])  
        ent_mention_surface_name_batch.append(record['ent_ment_surface_chars'])
        sentence_batch.append(record['sent_wds'])
        pos_batch.append(record['sent_pos'])
        ent_mention_surface_name_lent_batch.append(record['ent_ment_surface_chars_lent'])
        random_type_batch.append(type_)  
        random_ent_id.append(ent_id)  
        
        flag = ((ent_no+1) %batch_size==0 or (ent_no+1)== training_data_nums)
        
        
        if flag:
          if len(random_ent_mention_index_batch)!=0:
            assert(len(random_ent_mention_index_batch)==len(sentence_batch))
            
            yield pos_batch,random_ent_mention_index_batch,\
                    np.asarray(ent_mention_surface_name_batch,dtype=np.int32),\
                    np.asarray(ent_mention_surface_name_lent_batch,dtype=np.int32),\
                    np.asarray(sentence_batch),\
                    np.asarray(random_type_batch,np.float32),\
                    np.asarray(random_ent_id,np.int32)
          
          random_type_batch =[]
          random_ent_mention_index_batch =[]
          ent_mention_surface_name_batch=[]
          ent_mention_surface_name_lent_batch=[]
          sentence_batch=[]
          random_ent_id = []
          complete_sentence_batch=[]
          complete_pos_batch = []

        
    
  def get_shuffle_train_data_1(self,batch_size,training_data_nums,collection):
    
    for i in range(int(training_data_nums/batch_size)):
      ent_no=-1
      random_type_batch =[]
      random_ent_mention_index_batch =[]
      ent_mention_surface_name_batch=[]
      ent_mention_surface_name_lent_batch=[]
      sentence_batch=[]
      random_ent_id = []
      #we utilize collection aggregate sample to speed up the shuffle operations..
      for record in collection.aggregate([{"$sample": {"size": batch_size}}]):
        ent_id = record['ent_id']
        ent_mention_index = record['ent_index']
        type_ = record['type_']
        
        assert(np.sum(type_)!=0)
        ent_no  += 1
        random_ent_mention_index_batch.append([ent_no,ent_mention_index[0],ent_mention_index[1]])  
        ent_mention_surface_name_batch.append(record['ent_ment_surface_chars'])
        sentence_batch.append(record['sent_wds'])
        ent_mention_surface_name_lent_batch.append(record['ent_ment_surface_chars_lent'])
        random_type_batch.append(type_)  
        random_ent_id.append(ent_id)
  
        assert(len(random_ent_mention_index_batch)==len(sentence_batch))
        
      yield random_ent_mention_index_batch,\
              np.asarray(ent_mention_surface_name_batch,dtype=np.int32),\
              np.asarray(ent_mention_surface_name_lent_batch,dtype=np.int32),\
              np.asarray(sentence_batch),\
              np.asarray(random_type_batch,np.float32),\
              np.asarray(random_ent_id,np.int32)
                
      
        
      