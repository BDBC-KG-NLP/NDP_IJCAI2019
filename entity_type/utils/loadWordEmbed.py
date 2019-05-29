# -*- coding: utf-8 -*-
"""
@function: load pre-train word embedding, we remove the word that not in the vocabs.
"""
from tqdm import tqdm
import cPickle
import numpy as np
import tensorflow as tf

class wordEmbedding(object):
  def __init__(self,tag,args):
    self.word_embed_matrix = [] 
    self.vocab = {}
    
    #file_ = open('data/glove.840B.300d.txt')
    ids = 0
    self.vocab['#PAD#'] = ids
    self.word_embed_matrix.append(np.zeros((300,)))
    with open(args.dir_path+args.datasets+'/word_embedding.txt') as file_:
      for line in tqdm(file_):
        ids += 1
        if tag=='test':
          if ids ==10000:
            break
          
        split = line.split(" ")
        word = split[0]
        vector_strings = split[1:]
        self.word_embed_matrix.append(map(float,vector_strings))
        self.vocab[word] = ids
        
    
    ids += 1
    self.vocab['#UNK#'] = ids
    self.randomVector = cPickle.load(open('data/randomVector.p','rb'))
    self.word_embed_matrix.append(self.randomVector)
    self.vocab_size = ids + 1
    self.id2vocab = {self.vocab[key]:key for key in self.vocab}
    
    
  def is_in_vocabs(self,word):
    if word in self.vocab:
      return True
    elif word.lower() in self.vocab:
      return True
    else:
      return False
    
  def get_id_2_vocab(self,ids):
    return self.id2vocab[ids]
  
  def get_vocab_id(self,word):
    flag = True
    temp =None
    if word in self.vocab:
      temp = self.vocab[word]
    elif word.lower() in self.vocab:
      temp = self.vocab[word.lower()]
    else:
      temp = self.vocab['#UNK#']   #针对不同的数据处理，我感觉有很大的差异哈！
      
      
    #temp = self.randomVector   #otherwise, there are maybe something wrong to count the length!
            
    return flag,temp


if __name__ == '__main__':
  bbn_words = ['company','companies','co.','share','shares','stock','stocks','said','say']
  onto_words = ['year','years','say','said','company','companies','Japan','Japanese']
  wiki_words = ['state','states','year','years','play','plays','played']
  
  wd_2_id = {}
  dataset = 'Wiki'
  print(dataset)
  flags = tf.app.flags
  flags.DEFINE_string("dir_path",'data/KDD/',"data path")
  flags.DEFINE_string("datasets",dataset,"data")
  
  word_pairs =None
  if dataset == 'BBN':
    word_pairs = bbn_words
  elif dataset == 'OntoNotes':
    word_pairs = onto_words
  else:
    word_pairs = wiki_words
    
  args = flags.FLAGS
  
  wordEmbed = wordEmbedding('train',args)
  for wd in word_pairs:
    flag,ids = wordEmbed.get_vocab_id(wd)
    if wd not in wd_2_id:
      wd_2_id[wd] = ids
  print(wd_2_id)               
  cPickle.dump(wd_2_id,open(args.dir_path+dataset+'/wd_2_id_'+dataset+'.p','wb'))
  