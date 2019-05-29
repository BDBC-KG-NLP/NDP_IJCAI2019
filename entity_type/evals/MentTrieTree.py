# -*- coding: utf-8 -*-


class MentTrieTree(object):
  def __init__(self):
    self.root = {}
    self.stack=''

  def add(self, type_):
    word = type_.split('/')[1:]
    
    tree = self.root
    
    for char in word:
      sub_type_ = '/' + char
      if sub_type_ not in tree:
        tree[sub_type_] = {}
      tree = tree[sub_type_]

    tree['exist'] = True 
    
  def __iter__(self,input_dict=None):
    if not input_dict:
      input_dict=self.root
    
    if 'exist' in input_dict:
      yield (self.stack,input_dict['exist'])
    
    keys = [x for x in input_dict.keys() if x != 'exist']
    for key in keys:
      self.stack = self.stack+key
      for item in self.__iter__(input_dict[key]):
        yield item
      self.stack = self.stack[:-1]
    
    
  def search(self, type_):
    word = type_.split('/')[1:]
    tree = self.root
    
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
    
  def is_leaf(self,type_):
    word = type_.split('/')[1:]
    tree=self.root
    for char in word:
      sub_type_ = '/'+char
      if sub_type_ in tree:
        tree=tree[sub_type_]
      else:
        return False
    
    if 'exist' in tree and len(tree)==1:
      return True
    else:
      return False
    
  
if __name__ =='__main__':
  type_list = ['/wujs/t1','/wujs/t2','/wujs','/location','/wujs/t2/t3']
  
  type_tree = MentTrieTree()
  for type_ in type_list:
    type_tree.add(type_)
  
  print(type_tree.root)
  
  for ti in type_list:
    isleaf = type_tree.is_leaf(ti)
    print(ti, isleaf)