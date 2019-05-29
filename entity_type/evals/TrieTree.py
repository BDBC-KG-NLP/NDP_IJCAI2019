# -*- coding: utf-8 -*-


import cPickle
import numpy as np
import random
from MentTrieTree import MentTrieTree


def get_leaf_type(id2type,type2id,enti_all_types):
  tree = MentTrieTree()
  type_name_list= []
  for  ti in enti_all_types:
    tname = id2type[ti]
    tree.add(tname)
    type_name_list.append(tname)
  
  leaf_node_list=[]
  for tname in type_name_list:
    if tree.is_leaf(tname):
      tname_id = type2id[tname]
      
      leaf_node_list.append(tname_id)
  return leaf_node_list
          
class TrieTree(object):
  def __init__(self,artifical_noise_weight,dataset,type2id):
    self.tree = {}
    
    self.exist_num = 0
    self.dataset = dataset
    self.type2id = type2id
    self.type_num = len(self.type2id)
    self.artifical_noise_weight = artifical_noise_weight
    self.id2type = {self.type2id[i]:i for i in self.type2id}
    
    print(self.id2type[13])
    
    self.hieral_layer=2
    if dataset=='Wiki':
      self.hieral_layer=2
      self.minor_id2type = cPickle.load(open('data/KDD/Wiki/minor_id2type.p'))
    elif dataset == 'OntoNotes':
      self.hieral_layer=3
    elif dataset =='BBN':
      self.hieral_layer=2
    elif dataset =='knet':
      self.hieral_layer=2
      
    print('type_path')
    self.l1_type2id = {}
    self.l2_type2id = {}
    self.l3_type2id = {}
    
    self.l1_typeid = {}
    self.l2_typeid={}
    self.l3_typeid ={}
   
    self.artifical_ent_dict = {}
    self.type_parents={}
    self.type2typelist={}
    for key in self.type2id: 
      type_list = key.split('/')[1:]
      type_id_path = []
      if len(type_list)==1:
        self.l1_type2id[key] = [key]
        self.l1_typeid[self.type2id[key]]=[self.type2id[key]]
        
      elif len(type_list)==2:
        l1_type_ids = self.type2id['/' +'/'.join(type_list[0:1])]
        self.l2_type2id[key] = ['/' +'/'.join(type_list[0:1]),
                                '/' +'/'.join(type_list[0:2])]
        if l1_type_ids not in self.l2_typeid:
          self.l2_typeid[l1_type_ids] = []
          
        self.l2_typeid[l1_type_ids].append(self.type2id[key])
        if self.type2id[key] not in self.type_parents:
          self.type_parents[self.type2id[key]] = set()
        self.type_parents[self.type2id[key]].add(l1_type_ids)
        
      else:
        l2_type_ids = self.type2id['/' +'/'.join(type_list[0:2])]
        self.l3_type2id[key] = ['/' +'/'.join(type_list[0:1]),
                                 '/' +'/'.join(type_list[0:2]),
                                 '/' +'/'.join(type_list[0:3])]
        
        if l2_type_ids not in self.l3_typeid:
          self.l3_typeid[l2_type_ids]=[]

        self.l3_typeid[l2_type_ids].append(self.type2id[key])
        
        if self.type2id[key] not in self.type_parents:
          self.type_parents[self.type2id[key]] = set()
        self.type_parents[self.type2id[key]].add(l2_type_ids)
        
      
      for i in range(len(type_list)):
        rel_type = '/' +'/'.join(type_list[0:i+1])
        if rel_type in self.type2id:
          ids =  self.type2id[rel_type]
          type_id_path.append(ids)
          
      self.type2typelist[key] = type_id_path
    '''
    for k1 in self.l1_typeid:
      print(k1,self.id2type[k1])
      if k1 in self.l2_typeid:
        print(len(self.l2_typeid[k1]))
        for k2 in self.l2_typeid[k1]:
          print(k2,self.id2type[k2])
        print('---------------')'''
    assert(len(self.l1_type2id)+len(self.l2_type2id)+len(self.l3_type2id)==self.type_num) 
    
    self.all_child_type_list = self.type_parents.keys()
    random.shuffle(self.all_child_type_list)
    print('all_child_type:',self.all_child_type_list)
    
    self.typeToken2id,self.stop_token_ids = self.get_type_token()
    print('typeToken2id nums:',len(self.typeToken2id))
    print('self.stop_token_ids:',len(self.stop_token_ids))
    self.typeToken_num = len(self.typeToken2id)
    
    if self.hieral_layer==2:  
      assert(self.typeToken_num==len(self.l1_type2id)*2+len(self.l2_type2id))
    elif self.hieral_layer==3:
      assert(self.typeToken_num==len(self.l1_type2id)*3+len(self.l2_type2id)*2+len(self.l3_type2id))
      
    self.type_path = self.get_type_2_fully_type()
    print('self.type_path num:',len(self.type_path))
    print('------------------')
  
  #is the parameters..
  def create_prior(self,alpha=1.0):
    print('alpha:',alpha)
    prior = np.zeros((self.type_num,self.type_num))
    for x in self.id2type:
      type_name = self.id2type[x]
      tmp = np.zeros(self.type_num)
      tmp[x]=1.0
      
      for p_x in self.type2typelist[type_name]:
        if p_x!=x:
          tmp[p_x]=alpha
      prior[x,:]=tmp
      
    return prior
  
  def gen_struct_pair(self):
    type_set = set(list(range(self.type_num)))
      
    child_types =self.all_child_type_list
    child_pos_neg_types=[]
    child_pos_neg_types_tag=[[1]+[0]*20]*len(self.all_child_type_list)
    for child_ti in child_types:
      pos_i = [random.choice(list(self.type_parents[child_ti]))]
      
      others=type_set - self.type_parents[child_ti]
      pos_i += random.sample(others,20)
      child_pos_neg_types.append(list(pos_i))
      
    return np.expand_dims(np.array(child_types,np.int32),-1),\
           np.array(child_pos_neg_types,np.int32),\
           np.array(child_pos_neg_types_tag,np.int32)
    
  def get_type_token(self):
    stop_token_ids = []
    
    typeToken2id = {}
    tid = 0
    if self.hieral_layer==2:
      for ti in self.l1_type2id:
        typeToken2id[ti]=tid
        tid += 1 
        typeToken2id[ti+'_s'] = tid
        stop_token_ids.append(tid)
        tid += 1
      for ti in self.l2_type2id:
        typeToken2id[ti]=tid
        tid += 1
    elif self.hieral_layer==3:
      for ti in self.l1_type2id:
        typeToken2id[ti]=tid
        tid += 1 
        typeToken2id[ti+'_s'] = tid
        stop_token_ids.append(tid)
        tid += 1
        typeToken2id[ti+'_s_s'] = tid
        stop_token_ids.append(tid)
        tid += 1
  
      for ti in self.l2_type2id:
        typeToken2id[ti]=tid
        tid += 1
        typeToken2id[ti+'_s'] = tid
        stop_token_ids.append(tid)
        tid += 1
      for ti in self.l3_type2id:
        typeToken2id[ti]=tid
        tid += 1
        
    return typeToken2id,stop_token_ids 

  def get_type_2_fully_type(self):
    #build the complete path
    full_type = np.ones((self.type_num,self.hieral_layer),dtype=np.int32)*len(self.typeToken2id)
    for type_id in range(self.type_num):
      type_name = self.id2type[type_id]
      
      type_id_path = []
      
      if self.hieral_layer==2:
        if type_name in self.l1_type2id:
          type_id_path.append(self.typeToken2id[type_name])
          type_id_path.append(self.typeToken2id[type_name+'_s'])
        
        elif type_name in self.l2_type2id:
          for typei in self.l2_type2id[type_name]:   #we exist something wrong here!
            type_id_path.append(self.typeToken2id[typei])
        else:
          print(self.dataset,' has wrong type paths..')
      elif self.hieral_layer==3:
        if type_name in self.l1_type2id:
          type_id_path.append(self.typeToken2id[type_name])
          type_id_path.append(self.typeToken2id[type_name+'_s'])
          type_id_path.append(self.typeToken2id[type_name+'_s_s'])
        elif type_name in self.l2_type2id:
          type_id_path.append(self.typeToken2id[self.l2_type2id[type_name][0]])
          type_id_path.append(self.typeToken2id[self.l2_type2id[type_name][1]])
          type_id_path.append(self.typeToken2id[self.l2_type2id[type_name][1]+'_s'])

        elif type_name in self.l3_type2id:  
          type_id_path.append(self.typeToken2id[self.l3_type2id[type_name][0]])
          type_id_path.append(self.typeToken2id[self.l3_type2id[type_name][1]])
          type_id_path.append(self.typeToken2id[self.l3_type2id[type_name][2]])
          
      for ids in range(len(type_id_path)):
        full_type[type_id][ids] = type_id_path[ids]
        
    return full_type
    
  def get_aritifical_noise_type(self,ent_rel_index,train_out):
    new_train_out = []
    for i in range(len(ent_rel_index)):
      ent_id = ent_rel_index[i]
      
      if ent_id in self.artifical_ent_dict:
        tag = [0]* self.type_num
        revise_type_list = self.artifical_ent_dict[ent_id]['revise_type_list']
        for ti in revise_type_list:
          tag[ti] = 1
        new_train_out.append(list(tag))
      else:
        new_train_out.append(train_out[i])  
    assert(len(new_train_out)==len(train_out))
    return new_train_out
  
  def get_denoise_result(self,filter_threshold,ent_rel_index,train_out,pred,pred_score):
    right_decode= 0.0
    for i in range(len(ent_rel_index)):
      ent_id = ent_rel_index[i]
      if ent_id in self.artifical_ent_dict:
        right_type_list = self.artifical_ent_dict[ent_id]['right_type_list']
        
        revise_type_list = self.artifical_ent_dict[ent_id]['revise_type_list']
        
        new_revise_type_list = set()
        
        
        for ti in revise_type_list:
          if pred_score[i][ti] > filter_threshold:
            new_revise_type_list.add(ti)
        if new_revise_type_list == right_type_list:
          right_decode += 1
    return right_decode
      
  def get_new_gold_type(self,ent_id,filter_threshold,gold_tag,train_tag,pred,pred_score):
    '''
    @#model has no prior information to tell us this kind of information
    '''
    type_path_nums = [0]*self.hieral_layer
    
    flag = False
    for typei in gold_tag:
      type_name = self.id2type[typei]
      type_path = type_name.split('/')[1:]
      type_path_nums[len(type_path)-1]+=1
      
      if type_path_nums[len(type_path)-1] >=2:
        flag =  True   #whether the data is the clean entity mention
        break
    '''
    @revise: 2018-10-23
    '''
    #type_list = list(train_tag)
    type_list=get_leaf_type(self.id2type,self.type2id,train_tag)
    '''
    type_pred = []
    for ti in type_list:
      type_pred.append(pred_score[ti])'''
    type_pred = pred_score[type_list]
      
    org_ret_type_set = set()
    revise_type_list = set(type_list)
    
    ti_min = np.argmin(type_pred)
    ti = type_list[ti_min]
    if pred_score[ti] < filter_threshold:
      #we also need to delete this node and its leaf nodes
      revise_type_list.remove(ti)
      
    if len(revise_type_list)==0:
      org_ret_type_set = set(type_list)  #ensure the type set is not null
    else:
      org_ret_type_set = revise_type_list
      
    '''
    @revise: 2018-10-23 we need to add the parent types
    '''
    ret_type_set = set(org_ret_type_set)
    
    for ti in org_ret_type_set:
      ret_type_set = ret_type_set | set(self.type2typelist[self.id2type[ti]])
    
    assert(len(ret_type_set)!=0) 
    
    right_2_wrong = 0.0
    #The right mention becomes wrong
    if flag == True:
      if ret_type_set != set(gold_tag):
        right_2_wrong += 1
    
    arti_non_delete=0.0;arti_right_delete=0.0;arti_wrong_delete=0.0
    if self.artifical_noise_weight!=0.0:
      if ent_id in self.artifical_ent_dict:
        right_type_set = set(self.artifical_ent_dict[ent_id]['right_type_list'])
        revise_type_set = set(self.artifical_ent_dict[ent_id]['revise_type_list'])
        
        #del_types = revise_type_set - ret_type_set
        if ret_type_set == revise_type_set: #do not delete the noise
          arti_non_delete += 1
        else:
          if ret_type_set == right_type_set:
            arti_right_delete += 1
          else:
            arti_wrong_delete += 1
        
    return right_2_wrong,arti_non_delete,arti_wrong_delete,arti_right_delete,ret_type_set


  
  def is_noise(self,type_list):
    type_path_nums = [0]*self.hieral_layer
    type_path_name = []
    
    for typei in type_list:
      type_name = self.id2type[typei]
      type_path = type_name.split('/')[1:]
      type_path_nums[len(type_path)-1]+=1
      type_path_name.append(type_name)
      
      if type_path_nums[len(type_path)-1] >=2:
        return True
    
    return False
  
  def is_noise_data(self,type_list):
    type_set = set()
    type_path_nums = [0]*self.hieral_layer
    type_path_name = []
    flag = False
    for typei in type_list:
      type_name = self.id2type[typei]
      type_path = type_name.split('/')[1:]
      type_path_nums[len(type_path)-1]+=1
      type_path_name.append(type_name)
      
      if type_path_nums[len(type_path)-1] >=2:
        flag =  True
      type_set.add(type_name)
      
    if flag == True:
      rand_id = random.choice(type_list)
      return self.type2typelist[self.id2type[rand_id]]
    else:
      return type_list
    
     
  def add(self, type_):
    word = type_.split('/')[1:]
    
    tree = self.tree
    
    for char in word:
      sub_type_ = '/' + char
      if sub_type_ not in tree:
        tree[sub_type_] = {}
      tree = tree[sub_type_]

    tree['exist'] = True 
    
  def DFS1(self,tree,type_list):
    l1_type=self.l1_typeid.keys()
    l1_type_score = self.pred_score[l1_type]
    
    l1_pred = l1_type[np.argmax(l1_type_score)]
    
    if l1_pred not in self.l2_typeid:
      return [l1_pred]
  
    l2_type = self.l2_typeid[l1_pred]
    l2_type_score = self.pred_score[l2_type]
    
    l2_pred = l2_type[np.argmax(l2_type_score)]
    l2_pred_score = np.max(l2_type_score)
    
    if l2_pred_score < 0:  #negative score, we do not expand
      return [l1_pred]
    else:
      if l2_pred not in self.l3_typeid:
        return [l1_pred,l2_pred]
      
      l3_type = self.l3_typeid[l2_pred]
      l3_type_score = self.pred_score[l3_type]
      
      l3_pred = l3_type[np.argmax(l3_type_score)]
      l3_pred_score = np.max(l3_type_score)
      
      if l3_pred_score <0:
        return [l1_pred,l2_pred]
      else:
        return [l1_pred,l2_pred,l3_pred]
    
  def DFS(self,tree,type_list):
    for sub_type in tree:
      if sub_type =="exist":
        type_name = ''.join(type_list)
        rel_ids = self.type2id[type_name]
        i=0
        score = 0
        for ids in self.type2typelist[type_name]:
          if i ==0:
            score = self.pred_score[ids]
          else:
            #we do not know what is the reason??
            if self.dataset == 'BBN':
              type_score = self.pred_score[ids]#-0.05
            elif self.dataset == 'OntoNotes':
              type_score = self.pred_score[ids]#-0.15#-0.2
            elif self.dataset =='Wiki':
              if rel_ids in self.minor_id2type:
                type_score = self.pred_score[ids]-0.1#-0.2
              else:
                type_score = self.pred_score[ids]-0.3#-0.5 #we fixed this parameter by the validation test..
            score = score*(1+type_score)
          i += 1
        #print type_name, score
        self.pred_score_new[rel_ids] = score
                           
      else:
        type_list.append(sub_type)
        type_list = self.DFS(tree[sub_type],type_list)
        
    type_list = type_list[0:-1] 
    return type_list
  
  def getResult(self,model_type,pred_data,target_data,threshold=0.0):
    batch_size,_ = np.shape(pred_data)
    
    batch_pred_type = []
    batch_pred_type_score = []
    batch_target_type = []
    for i in range(batch_size):
      self.pred_score = pred_data[i]
      #or model_type=='ShimaokeModel' or model_type=='ComplExClassificationModel'
      if model_type == 'ShimaokeModel':
        #threshold=0.5
        final_type = set()
        self.pred_score_new = pred_data[i]
        for si in range(len(self.pred_score)):
          if self.pred_score[si]>threshold:
            final_type = final_type | set(self.type2typelist[self.id2type[si]])
            #final_type.add(si)
      elif model_type == 'AbhishekModel' or model_type=='DenoiseAbhishekModel':
        self.pred_score_new = [0]*self.type_num
        final_type=set(self.DFS1(self.tree,[]))
      else:
         #if model_type == 'FullHierEnergyRandom' or model_type == 'FullHierEnergyRandomContext' or model_type=='PengModel' or  model_type=='ComplExClassificationModel':
        max_type_id = np.argmax(pred_data[i])
        self.pred_score_new = pred_data[i]
        max_type = self.id2type[max_type_id]
        final_type = set(self.type2typelist[max_type])
        
#      else: #our other models
#        self.pred_score_new = [0]*self.type_num
#        self.DFS(self.tree,[])
#        max_type_id = np.argmax(self.pred_score_new)
    
      batch_pred_type.append(final_type)
      batch_pred_type_score.append(self.pred_score_new)
      batch_target_type.append(set(np.nonzero(target_data[i])[0]))

    return batch_pred_type,batch_pred_type_score,batch_target_type
      
  def search(self, type_):
    word = type_.split('/')[1:]
    tree = self.tree
    
    for char in word:
      sub_type_ = '/'+char
      if sub_type_ in tree:
        tree = tree[sub_type_]
      else:
        
        return False

    if "exist" in tree and tree["exist"] == True:
      return True
    else:
      return False
  
if __name__ =='__main__':
  t = cPickle.load(open('../type2id.p'))['type2id']
  id2type = {t[typei]:typei for typei in t}
  
  type_tree = TrieTree('',t)
  for type_ in t:
    type_tree.add(type_)
  
  
