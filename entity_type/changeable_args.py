# -*- coding: utf-8 -*-

import tensorflow as tf
import cPickle
from evals import TrieTree
from utils import wordEmbedding,CharacterEmbedding
from pymongo import MongoClient

flags = tf.app.flags
flags.DEFINE_integer("epochs",20,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",1000,"batch size of training")
flags.DEFINE_integer("init_epoch",0,"we radom run this model for 5 iteration")
flags.DEFINE_integer("iterateEpoch",5,"iterateEpoch")
flags.DEFINE_float("p",0.1,"choice the root type as the negative type...")
flags.DEFINE_boolean("is_test",False,"whether is test...")
flags.DEFINE_boolean("is_training",True,"whether is test...")
flags.DEFINE_boolean("is_add_fnode",False,"whether is test...")
flags.DEFINE_integer("max_pos_type_l1",5,"max_pos_type_l1")
flags.DEFINE_integer("max_pos_type_l2",5,"max_pos_type_l2")
flags.DEFINE_integer("max_pos_type_l3",5,"max_pos_type_l3")
flags.DEFINE_integer("max_pos_type",5,"max_pos_type")
flags.DEFINE_integer("max_neg_type",5,"max_neg_type")
flags.DEFINE_integer("pos_dims",20,"pos_dims")
flags.DEFINE_string("version_no",'version2',"version_no")
flags.DEFINE_string("model_type",'EnergyRandom',"[EnergyRandom,]")
flags.DEFINE_integer("type_dim",100,"type_dim")
flags.DEFINE_float("margin",0.1,"margin")
flags.DEFINE_float("learning_rate",0.02,"['OntoNotes:0.02, ']learning rates")
flags.DEFINE_float("l2_loss_w",0.01,"l2_loss_w [0.01 for OntoNotes]")
flags.DEFINE_string("datasets",'OntoNotes',"[Wiki, OntoNotes, BBN]dataset name")
flags.DEFINE_string("dir_path",'data/KDD/',"dir_path")
flags.DEFINE_float("threshold",0.0,"threshold")
flags.DEFINE_integer("sentence_length",250,"max sentence length['OntoNotes':250, ]")
flags.DEFINE_integer("ctx_length",10,"batch size of training")  #we need to change?
flags.DEFINE_integer("class_size",89,"number of classes['OntoNotes':89, ]")
flags.DEFINE_integer("rnn_size",100,"hidden dimension of rnn['OntoNotes:100']")
flags.DEFINE_integer("characterLent",20,"characterLent")
flags.DEFINE_integer("attention_hidden_size",20,'attention_hidden_size')
flags.DEFINE_integer("cls_l1_size",100,'cls_l1_size')
flags.DEFINE_integer("char_rnn_size",200,"char_rnn_size")
flags.DEFINE_integer("word_dim",300,"word_dim")
flags.DEFINE_integer("feature_length",30,"feature_length")
flags.DEFINE_integer("num_layers",2,"number of layers in rnn")
flags.DEFINE_integer("adv2_dims",300,"adv2_dims")
flags.DEFINE_string("restore",'checkpoint/OntoNotes/',"path of saved model")
flags.DEFINE_string("log_dir",'logs/OntoNotes/',"path of saved model")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_boolean("save_intermediate",False,"apply dropout during training")
flags.DEFINE_boolean("use_clean",False,"weather utilize use clean")
flags.DEFINE_integer("iterate_num",0,"iterate_num")
flags.DEFINE_float("filter_threshold",0.5,"filter_threshold")
flags.DEFINE_float("artifical_noise_weight",0.0,"artifical_noise_weight")
flags.DEFINE_float("alpha",0.7,"artifical_noise_weight")
flags.DEFINE_float("keep_prob",0.5,"artifical_noise_weight")


args = flags.FLAGS

type2id = cPickle.load(open(args.dir_path + args.datasets+'/type2id.p'))['type2id']
id2type = {type2id[key]:key for key in type2id}

type_tree = TrieTree(args.artifical_noise_weight,args.datasets,type2id)
for type_ in type2id:
  type_tree.add(type_)

word2vecModel = wordEmbedding('train',args)
vocab_size = word2vecModel.vocab_size
characterEmbed = CharacterEmbedding()

client = MongoClient('mongodb://192.168.3.196:27017')
db = client['entity_typing'] # database name 

train_collection = db[args.datasets+'_train']
training_data_nums = train_collection.count()
