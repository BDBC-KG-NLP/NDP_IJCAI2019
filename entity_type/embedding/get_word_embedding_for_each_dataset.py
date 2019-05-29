# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:47:59 2017

@author: DELL
"""
from tqdm import tqdm
from collections import defaultdict

class wordEmbedForEachData(object):
  def load_stop_words(self):
    stop_w2id = {}
    with open('data/stopwords.txt') as file_:
      for line in file_:
        word = line.strip()
        if word not in stop_w2id:
          stop_w2id[word] = len(stop_w2id)
    return stop_w2id
        
  def get_words(self,word2freq,fName):
    with open(fName) as file_:
      for line in tqdm(file_):
        line = line.strip()
        
        items = line.split('\t')
        if len(items)>=1:
          word = items[0].lower() #全部小写啦
          word2freq[word] += 1
    return word2freq
  
  def load_subData(self,dir_path,dataset):
    word2freq = defaultdict(int)
    fName1 = dir_path + dataset+'/features/'+dataset+'Data_train.txt'
    fName2 = dir_path + dataset+'/features/'+dataset+'Data_testa.txt'
    fName3 = dir_path + dataset+'/features/'+dataset+'Data_testb.txt'
    
    word2freq = self.get_words(word2freq,fName1)
    print len(word2freq)
    word2freq = self.get_words(word2freq,fName2)
    print len(word2freq)
    word2freq = self.get_words(word2freq,fName3)
    print len(word2freq)
    return word2freq

  def __init__(self,tag,dir_path,dataset):
    self.word2id = {}
    self.word2id['NIL'] = 0
    self.stop_w2id = self.load_stop_words()
    print self.stop_w2id
    word2freq = self.load_subData(dir_path,dataset)
    '''
    filename = 'data/glove.840B.300d.txt'
    word2freq = self.load_subData(dir_path,dataset)
    print 'word2freq nums:',len(word2freq)
    
    output_file = open(dir_path+'/'+dataset+'/'+'word_embedding.txt','w')
    with open(filename) as file_:
      for line in tqdm(file_):
        row = line.strip().split(' ')
        word = row[0].lower()
        
        if word.lower() in word2freq:
          output_file.write(line)
          output_file.flush()
    output_file.close()'''
    #2018/3/12 we generate the top 50 words
    word2freq_sorted = sorted(word2freq.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    #we need to delete the stop words
    
    for key in word2freq_sorted[1:500]:
      word,freqs = key
      if word not in self.stop_w2id:
        print key
if __name__ =='__main__':
  dir_path = 'data/KDD/'
  tag = 'train'
  '''
  dataset = 'BBN'
  dataUtils = wordEmbedForEachData(tag,dir_path,dataset)
  
  
  dataset = 'OntoNotes'
  dataUtils = wordEmbedForEachData(tag,dir_path,dataset)
  '''
  
  
  dataset = 'Wiki'
  dataUtils = wordEmbedForEachData(tag,dir_path,dataset)
  
  '''
  dir_path = 'data/coarse_type/'
  dataset = 'nyt'
  dataUtils = wordEmbedForEachData(tag,dir_path,dataset)
  '''
  