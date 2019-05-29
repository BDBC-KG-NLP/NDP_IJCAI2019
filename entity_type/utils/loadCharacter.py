# -*- coding: utf-8 -*-

class CharacterEmbedding():
  def __init__(self):
    self.character2id = {}
    self.character2id['PAD'] = 0
    ids = 1
    with open('data/character.txt') as file_:
      for line in file_:
        line = line.strip()
        if line not in self.character2id:
          self.character2id[line] = ids 
          ids += 1
    
    self.character2id[' '] = ids
    ids += 1
    self.character2id['NIL'] = ids
    print 'ids:',ids 
    print 'character2id:',len(self.character2id)
                     
  def __getitem__(self,char):
    char = char.lower()
    if char in self.character2id:
      return self.character2id[char]
    else:
      return self.character2id['NIL']