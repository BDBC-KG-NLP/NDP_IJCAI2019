# -*- coding: utf-8 -*-

import numpy as np
'''
@sentence_final: shape:(batch_size,sequence_length,dims)
'''
def padZeros(sentence_final,max_sentence_length=80,dims=111):
  for i in range(len(sentence_final)):
    offset = max_sentence_length-len(sentence_final[i])
    sentence_final[i] += [[0]*dims]*offset
    
  return np.asarray(sentence_final)

def genEntCtxMask(args,batch_size,entment_mask_final):
  entNums = len(entment_mask_final)
  
  entCtxLeft_masks=[]
  entCtxLeft_pos=[]
  entCtxLeft_length=[]
  
  entCtxRight_masks=[]
  entCtxRight_pos=[]
  entCtxRight_length=[]
  for i in range(entNums):
    items = entment_mask_final[i]
    
    ids = items[0]
    assert(ids<batch_size) 
    start=items[1]
    end=items[2]
    
    temp_entCtxLeft_mask=[]
    temp_left_pos = []
    
    temp_entCtxRight_mask = []
    temp_right_pos = []
    '''
    @需要把start 和 end的信息加上啦！因为可能出现lent=0的情况啊！
    '''
    left = max(0,end-args.ctx_length); 
    right = min(args.sentence_length,start+args.ctx_length)
    
    left_lent = 0
    right_lent=0
    for ient in range(left,end):
      temp_left_pos.append(ient-end)
      temp_entCtxLeft_mask.append(ids*args.sentence_length+ient)
      
    for ient in range(start,right):
      temp_right_pos.append(ient-start)
      temp_entCtxRight_mask.append(ids*args.sentence_length+ient)
        
    if end-left < args.ctx_length:
      left_lent = end-left
      temp_entCtxLeft_mask+= [batch_size*args.sentence_length] * (args.ctx_length-(end-left))
      temp_left_pos +=[0]*(args.ctx_length-(end-left))
    else:
      left_lent = args.ctx_length
    
    if right-start < args.ctx_length:
      right_lent = right-start
      temp_entCtxRight_mask+= [batch_size*args.sentence_length] * (args.ctx_length-(right-start))
      temp_right_pos += [0]*(args.ctx_length-(right-start))
    else:
      right_lent = args.ctx_length
    
    entCtxRight_length.append(right_lent)
    entCtxLeft_length.append(left_lent)
    
    
    entCtxLeft_masks.append(temp_entCtxLeft_mask)
    
    entCtxLeft_pos.append(temp_left_pos)
    
    entCtxRight_masks.append(temp_entCtxRight_mask)
    entCtxRight_pos.append(temp_right_pos)

  return np.asarray(entCtxLeft_masks,np.int32),np.expand_dims(np.asarray(entCtxLeft_length,dtype=np.float32),-1),\
         np.expand_dims(np.asarray(entCtxLeft_pos,dtype=np.float32),-1),\
        np.asarray(entCtxRight_masks,np.int32),np.expand_dims(np.asarray(entCtxRight_length,dtype=np.float32),-1),\
         np.expand_dims(np.asarray(entCtxRight_pos,dtype=np.float32),-1)
    
def genEntMentMask(args,batch_size,entment_mask_final):
  entNums = len(entment_mask_final)
  entment_masks = []
  entment_length = []
  #need to limit the length of the entity mentions
  for i in range(entNums):
    entLent=5
    items = entment_mask_final[i]
    ids = items[0]
    start=items[1]
    end=items[2]
    assert(ids<batch_size)
    
    temp_entment_masks=[]
    for ient in range(start,end):
        temp_entment_masks.append(ids*args.sentence_length+ient)
        
    if end-start <5:
      entLent = end-start
      temp_entment_masks+= [batch_size*args.sentence_length] * (5-(end-start))
    elif end-start > 5:
      temp_entment_masks = temp_entment_masks[0:5]
    entment_length.append(entLent)
    
    entment_masks.append(list(temp_entment_masks))
  return np.asarray(entment_masks,dtype=np.int32),np.expand_dims(np.asarray(entment_length,dtype=np.float32),1)