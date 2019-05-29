# -*- coding: utf-8 -*-
"""
@functon: we generate all words that utilize to training...
"""
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from tqdm import tqdm

def get_word2freq(tag,word2freq):
  dir_path = 'data/KDD/knet/'
  ent_ments = np.load(dir_path+tag+'_entity.npy')
  ent_ctxs = np.load(dir_path+tag+'_context.npy')
  
  
  ent_nums = len(ent_ments)
  for i in tqdm(range(ent_nums)):
    ent_surface_name = ent_ments[i]
    ent_ctx = ent_ctxs[i]
    for word in ent_surface_name:
      if word not in ['unk']:
        if word not in word2freq:
          word2freq[word] =0
        word2freq[word] += 1
               
    for word in ent_ctx:
      if word not in ['unk']:
        if word not in word2freq:
          word2freq[word] =0
        word2freq[word] += 1
                 
  print 'word lents:',len(word2freq)
  return word2freq


word2freq = defaultdict(int)

word2freq = get_word2freq('test',word2freq)
word2freq = get_word2freq('manual',word2freq)
word2freq = get_word2freq('valid',word2freq)
word2freq = get_word2freq('train',word2freq)

filename = 'data/glove.840B.300d.txt'
word2id ={}
embeds =[]
wordId=0
with open(filename) as file_:
  for line in tqdm(file_.readlines()):
    row = line.strip().split(' ')
    word = row[0]
    word2id[word] = wordId
    embeds.append(' '.join(row[1:]))
    wordId += 1

new_word2id={}
new_embeds=[]     
new_ids =0

for wd in word2freq:
  rel_wd = None
  ids = None
  if wd in word2id:
    rel_wd = wd
  elif wd.lower() in word2id:
    rel_wd = wd.lower()
    
  if rel_wd != None:
    ids = word2id[rel_wd]
    
    if rel_wd not in new_word2id:
      new_word2id[rel_wd] = new_ids
      new_ids += 1
      new_embeds.append(embeds[ids])
    
    
fembed = open('data/KDD/knet/word_embedding.txt','w')

new_id2word={new_word2id[wd]:wd for wd in new_word2id}


for ids in range(len(new_id2word)):
  wd = new_id2word[ids]
  fembed.write(wd+' '+new_embeds[ids]+'\n')
  fembed.flush()
fembed.close()
  
  
    
    