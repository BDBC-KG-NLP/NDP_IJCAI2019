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


def get_id_in_new_sent(ids,ient,sent,input_data,batch_size,sentence_length):
  
  iwid = sent[ient]  
  
  if len(input_data) < sentence_length:  #a new sentence
    s_wid = len(input_data)
    if iwid not in input_data and iwid!=0: 
      input_data[iwid] = s_wid
      
  if iwid in input_data:
    word_id = ids*sentence_length + input_data[iwid]
  else:
    word_id = batch_size*sentence_length
    
  return input_data,word_id

def genEntCtxMask_new(args,batch_size,entment_mask_final,input_sents):
  sentence_length = args.feature_length
  
  entNums = len(entment_mask_final)
  entment_masks = []
  entment_length = []
  
  entCtxLeft_masks=[]
  entCtxLeft_pos=[]
  entCtxLeft_length=[]
  
  entCtxRight_masks=[]
  entCtxRight_pos=[]
  entCtxRight_length=[]
  
  ent_new_sent = []
  
  #need to limit the length of the entity mentions
  for i in range(entNums):
    
    input_data ={}  
    items = entment_mask_final[i]
    ids = items[0]
    
    sent = input_sents[ids] 
    
    start=items[1]
    end=items[2]
    
    assert(ids<batch_size)
    
    temp_entment_masks=[]
    for ient in range(start,min(start+5,end)):
      input_data,word_id = get_id_in_new_sent(ids,ient,sent,input_data,batch_size,sentence_length)
      temp_entment_masks.append(word_id)
      
    if min(end,start+5)-start<5:
      entLent = end-start
      temp_entment_masks+= [batch_size*sentence_length] * (5-(min(end,start+5)-start))
    
    entment_length.append(entLent)
    
    entment_masks.append(list(temp_entment_masks))
    
    '''
    @ctx
    '''
    temp_entCtxLeft_mask=[]
    temp_left_pos = []
    
    temp_entCtxRight_mask = []
    temp_right_pos = []
    
    #left = max(0,end-args.ctx_length); 
    #right = min(args.sentence_length,start+args.ctx_length)
    
    left = max(0,end-args.ctx_length); 
    right = min(len(sent),start+args.ctx_length)
    
    left_lent = 0
    right_lent=0
    for ient in range(left,end):
      temp_left_pos.append(ient-end)
      
      input_data,word_id = get_id_in_new_sent(ids,ient,sent,input_data,batch_size,sentence_length)
      assert(word_id<=batch_size*sentence_length)
      temp_entCtxLeft_mask.append(word_id)
    
    for ient in range(start,right):
      temp_right_pos.append(ient-start)
      input_data,word_id = get_id_in_new_sent(ids,ient,sent,input_data,batch_size,sentence_length)
      assert(word_id<=batch_size*sentence_length)
      temp_entCtxRight_mask.append(word_id)
        
    if end-left < args.ctx_length:
      left_lent = end-left
      temp_entCtxLeft_mask+= [batch_size*sentence_length] * (args.ctx_length-left_lent)
      temp_left_pos +=[0]*(args.ctx_length-left_lent)
    else:
      left_lent = args.ctx_length
    
    if right-start < args.ctx_length:
      right_lent = right-start
      temp_entCtxRight_mask+= [batch_size*sentence_length] * (args.ctx_length-right_lent)
      temp_right_pos += [0]*(args.ctx_length-right_lent)
    else:
      right_lent = args.ctx_length
    
    input_data_re = {val:key for key,val in input_data.iteritems()}
    input_sent_i = []
    lent_new_sent_i = len(input_data_re)
    for i_re in range(lent_new_sent_i):
      input_sent_i.append(input_data_re[i_re])
    
    if lent_new_sent_i <sentence_length:
      input_sent_i += [0]*(sentence_length-lent_new_sent_i)
      
    assert(len(input_sent_i)==30)
    ent_new_sent.append(input_sent_i)
    
    entCtxRight_length.append(right_lent)
    entCtxLeft_length.append(left_lent)
    
    
    entCtxLeft_masks.append(temp_entCtxLeft_mask)
    
    entCtxLeft_pos.append(temp_left_pos)
    
    entCtxRight_masks.append(temp_entCtxRight_mask)
    entCtxRight_pos.append(temp_right_pos)
  
  #2018/3/1 we need to revise our code!
  sent_size = len(ent_new_sent)
  if sent_size < batch_size: 
    #print sent_size
    #print np.shape(ent_new_sent)
    pad_data = np.asarray([[0]*args.feature_length]*(batch_size-sent_size))
    #print np.shape(pad_data)
    ent_new_sent = np.concatenate((ent_new_sent,pad_data),0)
  
  ent_new_sent = np.asarray(ent_new_sent,dtype=np.int32)
  
  #print 'ent_new_sent:',np.shape(ent_new_sent)
  return np.asarray(ent_new_sent,np.int32),np.asarray(entment_masks,dtype=np.int32),np.expand_dims(np.asarray(entment_length,dtype=np.float32),1),\
          np.asarray(entCtxLeft_masks,np.int32),np.expand_dims(np.asarray(entCtxLeft_length,dtype=np.float32),-1),\
         np.expand_dims(np.asarray(entCtxLeft_pos,dtype=np.float32),-1),\
        np.asarray(entCtxRight_masks,np.int32),np.expand_dims(np.asarray(entCtxRight_length,dtype=np.float32),-1),\
         np.expand_dims(np.asarray(entCtxRight_pos,dtype=np.float32),-1)