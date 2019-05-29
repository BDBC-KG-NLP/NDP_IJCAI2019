# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import cPickle
from utils import genEntCtxMask_new,gen_pos_neg_tag
from tqdm import tqdm
import model_args 

def main(_):
  args = model_args.args
  
  if args.artifical_noise_weight != 0.0:
    
    artifical_ent_dict = cPickle.load(open(args.dir_path + args.datasets+'/'+args.version_no+ '/generated/'+'artifical_ent_dict.p'+'_'
                    +str(args.artifical_noise_weight),'rb'))
  
    
    model_args.type_tree.artifical_ent_dict = artifical_ent_dict

    
  old_modesaveDataset = '_'.join(['artifical',str(args.artifical_noise_weight),args.datasets,args.model_type,str(args.learning_rate),
                             str(args.rnn_size),str(args.attention_hidden_size),
                             str(args.type_dim) ,str(args.margin),'sl2',str(args.l2_loss_w),
                             str(args.max_neg_type),str(args.max_pos_type),
                             str(args.is_add_fnode),
                             'r',str(args.iterate_num-1),
                             str(args.filter_threshold)
                    ])

  modesaveDataset = '_'.join(['artifical',str(args.artifical_noise_weight),args.datasets,args.model_type,str(args.learning_rate),
                             str(args.rnn_size),str(args.attention_hidden_size),
                             str(args.type_dim) ,str(args.margin),'sl2',str(args.l2_loss_w),
                             str(args.max_neg_type),str(args.max_pos_type),
                             str(args.is_add_fnode),
                             'r',str(args.iterate_num),str(args.filter_threshold)
                    ])
  
  
  last_new_gen_train_out = cPickle.load(open(args.dir_path + args.datasets+'/'+args.version_no+'/iter_generated/'+'new_gen_train_out.p_'+old_modesaveDataset,'rb'))
  
  model = model_args.model
  sess = model_args.sess
 
  new_gen_train_out = {}
  
  
  if model.load(sess,args.restore,modesaveDataset,model.tvars):
    print("[*] "+modesaveDataset +"is loaded...")
  else:
    print('no checkpoint: '+ modesaveDataset)
  increase_delete_num = 0.0
  arti_complete_right_all = 0.0
  arti_partial_right_all = 0.0
  arti_wrong_del_all = 0.0
  arti_non_del_all = 0.0
  right_2_wrong_all = 0.0
  train_datas =list(model_args.train_collection.find({}))
  for train_entment_mask,\
       train_entment_chars,train_entment_chars_lent,\
      train_sentence_final,train_tag_final,\
       ent_rel_index in tqdm(model_args.train_data_reader.get_shuffle_train_data(args.batch_size,
                                                                      model_args.training_data_nums,
                                                                      train_datas)):     
   
    train_input,train_entMentIndex,entMent_length,\
        train_entCtxLeft_Index,ctxLeft_lent,pos2,\
       train_entCtxRight_Index,ctxRight_lent,pos3 =  genEntCtxMask_new(args,args.batch_size,train_entment_mask,
                                                                    train_sentence_final) 
    
    #whether delete at the last around?
    new_train_out = []
    new_ent_rel_index = []
    new_ids = []
    
    for ids in range(len(ent_rel_index)):
      ent_id = ent_rel_index[ids]

      if ent_id in last_new_gen_train_out:
        new_ids.append(ids)
        new_ent_rel_index.append(ent_rel_index[ids]) #we need to take care of those parameters..
        
        type_set  = last_new_gen_train_out[ent_id] #very esay to do something wrong...
        new_train_out_i = [0] * args.class_size
        for ti in type_set:
          new_train_out_i[ti]=1
      
        new_train_out.append(new_train_out_i)
      else:
        print('ent not in filtered ...')
        exit(0)
      
    train_out = np.asarray(new_train_out,np.float32)
   
    feed_in_dict= {model.input_data:train_input,
                           model.entment_chars:train_entment_chars, model.entment_chars_lent:train_entment_chars_lent,
                           model.output_data:train_out,
                           model.entMentIndex:train_entMentIndex,
                           model.entCtxLeftIndex:train_entCtxLeft_Index,
                           model.entCtxRightIndex:train_entCtxRight_Index,
                           model.entMentLent:entMent_length,
                           model.ctxLeftLent:ctxLeft_lent,model.ctxRightLent:ctxRight_lent,
                           model.keep_prob:1.0}
        
    pred= sess.run(model.prediction, feed_in_dict)

    pred_re,pred_score_re,train_out_re = model_args.type_tree.getResult(model_args.args.model_type,pred,train_out,args.threshold)
    
    for i in range(len(train_out)):
      ent_id = ent_rel_index[i]
      i_gold_type = np.nonzero(train_tag_final[i])[0]
      i_train_type = train_out_re[i]
      
      right_2_wrong,arti_non_delete,\
      arti_wrong_delete,arti_right_delete,\
      new_train_out_i = model_args.type_tree.get_new_gold_type(ent_id,args.filter_threshold,
                                                          i_gold_type,i_train_type,pred_re[i],pred[i])
      right_2_wrong_all+=right_2_wrong
      arti_complete_right_all += arti_right_delete
      arti_partial_right_all += 0
      arti_wrong_del_all += arti_wrong_delete
      arti_non_del_all += arti_non_delete
      
      if set(new_train_out_i) != set(i_train_type):
        increase_delete_num +=1
#        print('new increase_delete_num:',increase_delete_num)
#        print(new_train_out_i)
#        print(i_train_type)
#        print('----------------------')
      if new_train_out_i !=None:
        new_gen_train_out[ent_id] = new_train_out_i
      else:
        print('type is none: totally delete')
        
  if args.artifical_noise_weight != 0.0:
    decode_complete_clean = arti_complete_right_all/len(artifical_ent_dict)
    decode_partial_clean = arti_partial_right_all/len(artifical_ent_dict)
    decode_wrong_del = arti_wrong_del_all/len(artifical_ent_dict)
    decode_non_del = arti_non_del_all/len(artifical_ent_dict)
    params = {'decode_complete_clean':decode_complete_clean,'decode_partial_clean':decode_partial_clean,
              'decode_wrong_del':decode_wrong_del,'decode_non_del':decode_non_del}
    
    cPickle.dump(params,open(args.dir_path + args.datasets+'/'+args.version_no+'/result/'+'decode_noise.p_'+modesaveDataset,'wb'))
  
  print('new increase_delete_num:',increase_delete_num)
  print('---------------------')
  assert(len(new_gen_train_out)==model_args.training_data_nums)
  cPickle.dump(new_gen_train_out,open(args.dir_path + args.datasets+'/'+args.version_no+'/iter_generated/'+'new_gen_train_out.p_'+modesaveDataset,'wb'))
  
  param_total = {'increase_delete_num':increase_delete_num/model_args.training_data_nums,'right2wrong':right_2_wrong_all/model_args.training_data_nums}
  
  cPickle.dump(param_total,open(args.dir_path + args.datasets+'/'+args.version_no+'/result/'+'right2wrong.p_'+modesaveDataset,'wb'))
  model_args.client.close()
  sess.close()
 
if __name__=='__main__':
  tf.app.run()