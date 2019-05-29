# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:57:21 2018

@author: wujs
"""
import tensorflow as tf

import numpy as np
import changeable_args
from embedding import NMTDataRead
import cPickle
#DenoiseAbhishekModel
from model import  FullHierEnergy,FullHierEnergyRandom,FullHierEnergyContext,\
                   ShimaokeModel,AbhishekModel,PengModel,\
                   ComplExModel,ComplExRandomModel,ComplExContextModel

args = changeable_args.args
characterEmbed = changeable_args.characterEmbed
word2vecModel = changeable_args.word2vecModel
type_tree = changeable_args.type_tree

client = changeable_args.client
db = changeable_args.db

train_data_reader =  NMTDataRead(args,characterEmbed,word2vecModel,args.datasets,'train')
train_collection = db[args.datasets+'_train']
training_data_nums = train_collection.count()

testb_data_reader = NMTDataRead(args,characterEmbed,word2vecModel,args.datasets,'testb')
test_data = testb_data_reader.get_input_figerTest_chunk(args.batch_size)
  
testa_data_reader = NMTDataRead(args,characterEmbed,word2vecModel,args.datasets,'testa')
valid_data = testa_data_reader.get_input_figerTest_chunk(args.batch_size)

type_correlation = cPickle.load(open(args.dir_path+args.datasets+'/total_type_correlation.p','rb'))

embed_data= np.asarray(word2vecModel.word_embed_matrix,np.float32)

config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=8,inter_op_parallelism_threads=8)
config.gpu_options.allow_growth=True

if args.model_type == 'ComplExModel':
  model = ComplExModel(changeable_args)
elif args.model_type == 'ComplExRandomModel':
  model = ComplExRandomModel(changeable_args)
elif args.model_type =='ComplExContextModel':
  model = ComplExContextModel(changeable_args)
elif args.model_type =='FullHierEnergy':
  model = FullHierEnergy(changeable_args)
elif args.model_type == 'FullHierEnergyRandom':
  model = FullHierEnergyRandom(changeable_args)
elif args.model_type == 'FullHierEnergyContext':
  model = FullHierEnergyContext(changeable_args)
elif args.model_type =='ShimaokeModel':
  model = ShimaokeModel(changeable_args)
elif args.model_type =='AbhishekModel':
  model = AbhishekModel(changeable_args)
#elif args.model_type =='DenoiseAbhishekModel':
#  model = DenoiseAbhishekModel(changeable_args)
elif args.model_type == 'PengModel':
  model = PengModel(changeable_args)
else:
  print('wrong model..')
  exit(0)
  
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer(),feed_dict={model.word_embed_pl:embed_data})
sess.run(tf.local_variables_initializer())