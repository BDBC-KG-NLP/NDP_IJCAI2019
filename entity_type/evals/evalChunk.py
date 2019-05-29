# -*- coding: utf-8 -*-
"""
function: chunk evaluation
"""
from tqdm import tqdm
import collections
import cPickle

#first chunk + Freebase alias
def openSentid2aNosNoid(dir_path,tag):
  sNo = 0
  sid2aNosNo = {}
  aNosNo2sid = {}
  
  fName = dir_path +'process/'+tag+'_sentid2aNosNoid.txt'

  for line in open(fName):
    line = line.strip()
    
    sid,aNosNo = line.split('\t')
    sid2aNosNo[sNo] = aNosNo
    aNosNo2sid[aNosNo] = sNo
    sNo += 1
  return sid2aNosNo,aNosNo2sid

def getaNosNo2entMenOntos(dir_path,tag):
  param = cPickle.load(open(dir_path+'type2id.p','rb'))
  type2id = param['type2id']
  totalEnts = 0
  #lists: [[entSatrt,entEnd],...]
  aNosNo2entMen = collections.defaultdict(list)
  fName = dir_path +'process/'+tag+'_gold_entMen2aNosNoid.txt'
  for line in open(fName):
    line = line.strip()
    items = line.split('\t')
    
    ents = items[1];ente = items[2]
    aNosNo = items[3]
    typeIdList=[]
    typeList=[]
    #flag = False
    flag = False
    totalEnts += 1
    type_list = items[4:]
    
    type_num  =  len(type_list)
    for typei in type_list:
      if typei in type2id:
        typeIdList.append(type2id[typei])
        typeList.append(typei)
        flag = True
    #print typeList
#    if flag == False:
#      for typei in ["/other", "/location"]:
#        if typei in type2id:
#          typeList.append(type2id[typei])
#          fflag=True
          
    if flag:
      aNosNo2entMen[aNosNo].append([ents,ente,typeIdList,items[0]])   
    else:
      print 'no mapping type:',line
  return aNosNo2entMen
