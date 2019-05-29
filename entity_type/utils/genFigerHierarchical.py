# -*- coding: utf-8 -*-


import cPickle
import numpy as np
from sklearn.preprocessing import normalize

def genTypePrior(type2id,entMents):
  ret = np.zeros((len(type2id,)),dtype=np.float32)

  for i in range(len(entMents)):
    entList = entMents[i]
    for enti in entList:
      for typei in enti[2]:
        ret[typei] += 1
  return normalize(ret)

def generateFigerHier():
  figer2id = cPickle.load(open('data/figer/figer2id.p','rb'))
  id2figer = {figer2id[key]:key for key in figer2id}
  
  print len(figer2id)
  #get the first level
  row = 0
  first2Level = {}
  secondLevel = []
  for figer in figer2id:
    if len(figer.split('/')) ==2:
      first2Level[figer] = [row]
      row += 2
    else:
      secondLevel.append(figer)
  
  second2Level={}        
  for figer in secondLevel:
    hasHier=False
    for key in first2Level:
      if key in figer:
        hasHier=True
        rowId = first2Level[key][0]
        second2Level[figer] =[rowId,rowId+1]
    if hasHier==False:
      second2Level[figer]=[row]
      row += 1
        
  figerHier = np.zeros((row,len(figer2id)))
  figer2level = dict(first2Level,**second2Level)
  
  for ids in id2figer:
    for row in figer2level[id2figer[ids]]:
      #print row,ids
      figerHier[row,ids]=1
               
  print np.shape(figerHier)             
  cPickle.dump(figerHier,open('data/figer/figerhierarchical.p','wb'))
  
def generateOntoNotesHier():
  type2id = cPickle.load(open('data/OntoNotes/type2id.p','rb'))
  id2type = {type2id[typei]:typei for typei in type2id}
  '''
  @has three levels
  '''
  row =0
  first2Level={}
  secondLevel =[]
  thirdLevel = []
  
  for typei in type2id:
    items = typei.split('/')
    if len(items)==2:
      first2Level[typei]=[row]
      row += 3
    elif len(items)==3:
      secondLevel.append(typei)
    elif len(items)==4:
      thirdLevel.append(typei)
    else:
      print 'wrong types ',typei
  print first2Level
  print '--------------'
  print secondLevel
  print '---------------'
  print thirdLevel
  print '---------------'
  second2Level={}
  for typei in secondLevel:
    hasHier=False
    for key in first2Level:
      if typei.startswith(key):
        hasHier=True
        rowId = first2Level[key][0]
        second2Level[typei] =[rowId,rowId+1]
    if hasHier==False:
      print typei,'has no the first layer...'
      secondLevel[typei]=[row]
      row += 3
      
  third2Level={}
  for typei in thirdLevel:
    hasHier=False
    for key in second2Level:
      if typei.startswith(key):
        hasHier=True
        rowId = second2Level[key][0]
        third2Level[typei]=[rowId,rowId+1,rowId+2]
    if hasHier==False:
      print typei,'has no the second layer...'
      third2Level[typei]=[row]
      row += 3
  
  OntoNotesHier = np.zeros((row,len(type2id)))
  OntoNotes2level = dict(first2Level,**dict(second2Level,**third2Level))
  
  for ids in id2type:
    for row in OntoNotes2level[id2type[ids]]:
      #print row,ids
      OntoNotesHier[row,ids]=1
  cPickle.dump(OntoNotesHier,open('data/OntoNotes/OntoNoteshierarchical.p','wb'))


def generateBBNHier():
  type2id = cPickle.load(open('data/KDD/BBN/type2id.p','rb'))
  id2type = {type2id[key]:key for key in type2id}
  
  print len(type2id)
  #get the first level
  row = 0
  first2Level = {}
  secondLevel = []
  for typei in type2id:
    if len(typei.split('/')) ==2:
      first2Level[typei] = [row]
      row += 2
    else:
      secondLevel.append(typei)
  
  second2Level={}        
  for typei in secondLevel:
    hasHier=False
    for key in first2Level:
      if key in typei:
        hasHier=True
        rowId = first2Level[key][0]
        second2Level[typei] =[rowId,rowId+1]
    if hasHier==False:
      second2Level[typei]=[row]
      row += 1
        
  typeHier = np.zeros((row,len(type2id)))
  type2level = dict(first2Level,**second2Level)
  
  for ids in id2type:
    for row in type2level[id2type[ids]]:
      #print row,ids
      typeHier[row,ids]=1
               
  print np.shape(typeHier)             
  cPickle.dump(typeHier,open('data/KDD/BBN/hierarchical.p','wb'))
  
  
#generateFigerHier()
#generateOntoNotesHier()
generateBBNHier()

#dir_path = 'data/figer/'
#type2id = cPickle.load(open(dir_path+'figer2id.p'))
#entMents =cPickle.load(open(dir_path+'features/train_entMents.p'))
#print entMents
#type2weight = genTypePrior(type2id,entMents)
#print type2weight
#cPickle.dump(type2weight,open(dir_path+'type2weight.p','wb'))

