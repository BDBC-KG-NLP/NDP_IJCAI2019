#we have split the dataste into training,test and validation
###########################################
#Insert data into mongoDB
###########################################
CUDA_VISIBLE_DEVICES=0 python insert_knet_2_mongo.py --is_test=True --datasets=BBN --batch_size=1000  --class_size=47 --rnn_size=100 --type_dim=50 --sentence_length=252

for model_type in FullHierEnergy,FullHierEnergyRandom,FullHierEnergyContext
do
{
  for artifical_noise_weight in 0.0
  do
  {
    CUDA_VISIBLE_DEVICES=0 python trainHierEnergy.py --is_test=False --is_training=True --datasets=BBN --batch_size=1000  --class_size=47 --rnn_size=100 --type_dim=100 --attention_hidden_size=40 --sentence_length=252 --iterateEpoch=5 --epochs=30 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.001 --margin=0.3 --max_neg_type=30 --max_pos_type=36 --max_pos_type_l1=6 --max_pos_type_l2=6 --restore=checkpoint/BBN --init_epoch=0  --artifical_noise_weight=${artifical_noise_weight} --version_no=version2 --model_type=${model_type} --is_add_fnode=False
  }
  done
}
done

#ComplEx model need much more iterations for training test...
for model_type in ComplExModel,ComplExRandomModel,ComplExContextModel
do
{
  for artifical_noise_weight in 0.0
  do
  {
    CUDA_VISIBLE_DEVICES=1 python trainShimaoka.py --is_test=False --is_training=True --datasets=BBN --batch_size=1000  --class_size=47 --rnn_size=100 --type_dim=50 --attention_hidden_size=40 --sentence_length=252 --iterateEpoch=5 --epochs=100 --learning_rate=0.001 --threshold=0.0 --l2_loss_w=0.001 --margin=0.3 --max_neg_type=30 --max_pos_type=36 --max_pos_type_l1=6 --max_pos_type_l2=6 --restore=checkpoint/BBN --init_epoch=0 --artifical_noise_weight=0.0 --version_no=version2 --model_type=${model_type} --use_clean=False
  
  }
  done
}
done


CUDA_VISIBLE_DEVICES=1 python trainShimaoka.py --is_test=False --is_training=True --datasets=BBN --batch_size=1000  --class_size=47 --rnn_size=100 --attention_hidden_size=20 --sentence_length=252 --iterateEpoch=20 --epochs=20 --learning_rate=0.01 --threshold=0.5 --l2_loss_w=0.0 --margin=0.3 --max_neg_type=30 --max_pos_type=6 --restore=checkpoint/BBN/version1 --type_dim=100 --init_epoch=0 --artifical_noise_weight=0.0 --version_no=version1 --model_type=ShimaokeModel --use_clean=False


CUDA_VISIBLE_DEVICES=0 python trainShimaoka.py --is_test=False --is_training=True --datasets=BBN --batch_size=1000  --class_size=47 --rnn_size=100 --type_dim=100 --attention_hidden_size=20 --sentence_length=252 --iterateEpoch=20 --epochs=20 --learning_rate=0.001 --threshold=0.0 --l2_loss_w=0.0 --margin=0.3 --max_neg_type=30 --max_pos_type=6 --restore=checkpoint/BBN/version1  --init_epoch=0 --artifical_noise_weight=0.0 --version_no=version1 --model_type=DenoiseAbhishekModel --use_clean=False


CUDA_VISIBLE_DEVICES=1 python trainShimaoka.py --is_test=False --is_training=True --datasets=BBN --batch_size=1000  --class_size=47 --rnn_size=100 --type_dim=100 --attention_hidden_size=20 --sentence_length=252 --iterateEpoch=20 --epochs=20 --learning_rate=0.001 --threshold=0.0   --l2_loss_w=0.0 --margin=0.3 --max_neg_type=30 --max_pos_type=6 --restore=checkpoint/BBN/version1 --init_epoch=0 --artifical_noise_weight=0.0 --version_no=version1 --model_type=DenoiseAbhishekModel --use_clean=True


python trainShimaoka.py --is_test=False --is_training=True --datasets=BBN --batch_size=1000  --class_size=47 --rnn_size=100 --type_dim=100 --attention_hidden_size=20 --sentence_length=252 --iterateEpoch=20 --epochs=40 --learning_rate=0.001 --threshold=0.0   --l2_loss_w=0.001 --margin=0.3 --max_neg_type=30 --max_pos_type=6 --restore=checkpoint/BBN/version1 --init_epoch=0 --artifical_noise_weight=0.0 --version_no=version1 --model_type=PengModel --use_clean=False --alpha=0.1


#we also provide iterative training process for our model.
#filter threshold: {0.0,0.1,-0.1,-0.2}
#iterative number:
#for filter_threshold in {0.0,0.1}
#do
#{
#
#  CUDA_VISIBLE_DEVICES=1 python gen_baseline_tag.py --is_test=False --datasets=BBN --batch_size=1000 --class_size=47 --rnn_size=100 --attention_hidden_size=40 --type_dim=50 --sentence_length=252 --iterateEpoch=1 --epochs=2 --learning_rate=0.001 --threshold=0.0   --l2_loss_w=0.001 --margin=0.3 --max_neg_type=30 --max_pos_type=36 --max_pos_type_l1=6 --max_pos_type_l2=6 --restore=checkpoint/BBN/version2 --init_epoch=0 --artifical_noise_weight=0.0 --filter_threshold=${filter_threshold} --iterate_num=0 --version_no=version2 --model_type=FullHierEnergyRandom
#
#  for iterate_num in {1,2,3,4,5,6,7,8,9,10}
#  do
#  {
#    CUDA_VISIBLE_DEVICES=1 python trainReHierEnergy.py --is_test=False --datasets=BBN --batch_size=1000  --class_size=47 --rnn_size=100 --type_dim=50 --attention_hidden_size=40 --sentence_length=252 --iterateEpoch=2 --epochs=2 --learning_rate=0.001 --threshold=0.0   --l2_loss_w=0.001 --margin=0.3 --max_neg_type=30 --max_pos_type=36 --max_pos_type_l1=6 --max_pos_type_l2=6 --restore=checkpoint/BBN/version2 --init_epoch=0  --artifical_noise_weight=0.0 --version_no=version2 --model_type=FullHierEnergyRandom  --iterate_num=${iterate_num} --filter_threshold=${filter_threshold} --is_add_fnode=False
#
#    CUDA_VISIBLE_DEVICES=1 python gen_iter_tag.py --is_test=False --datasets=BBN --batch_size=1000  --class_size=47 --rnn_size=100 --type_dim=50 --attention_hidden_size=40 --sentence_length=252 --iterateEpoch=2 --epochs=10 --learning_rate=0.001 --threshold=0.0   --l2_loss_w=0.001 --margin=0.3 --max_neg_type=30 --max_pos_type=36 --max_pos_type_l1=6 --max_pos_type_l2=6 --restore=checkpoint/BBN/version2 --init_epoch=0  --artifical_noise_weight=0.0 --version_no=version2 --model_type=FullHierEnergyRandom  --iterate_num=${iterate_num} --filter_threshold=${filter_threshold} --is_add_fnode=False
#   }
#   done
#}
#done
