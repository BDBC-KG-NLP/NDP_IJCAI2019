#we have split the dataste into training,test and validation
###########################################
#Insert data into mongoDB
CUDA_VISIBLE_DEVICES=0 python insert_knet_2_mongo.py --is_test=True --datasets=Wiki --batch_size=2000 --class_size=128 --rnn_size=300 --type_dim=200 --attention_hidden_size=100 --sentence_length=62


for model_type in FullHierEnergyContext,FullHierEnergy,FullHierEnergyRandom
do
{
  CUDA_VISIBLE_DEVICES=0 python trainHierEnergy.py --is_test=False --datasets=Wiki --batch_size=2000 --class_size=128 --rnn_size=300 --type_dim=200 --attention_hidden_size=100 --sentence_length=62 --iterateEpoch=1 --epochs=10 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.001 --margin=0.2 --max_neg_type=100 --max_pos_type=70 --max_pos_type_l1=10 --max_pos_type_l2=7  --restore=checkpoint/Wiki/version1 --init_epoch=0 --artifical_noise_weight=0.0 --iterate_num=0 --version_no=version1  --model_type=${model_type} 
}
done

for model_type in ComplExModel,ComplExRandomModel,ComplExContextModel
do
{
  CUDA_VISIBLE_DEVICES=1 python trainShimaoka.py --is_test=False --datasets=Wiki --batch_size=2000 --class_size=128 --rnn_size=300 --type_dim=200 --attention_hidden_size=100 --sentence_length=62 --iterateEpoch=1 --epochs=10 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.001 --margin=0.2 --max_neg_type=100 --max_pos_type=70 --max_pos_type_l1=10 --max_pos_type_l2=7  --restore=checkpoint/Wiki/version1 --init_epoch=0 --artifical_noise_weight=0.0 --iterate_num=0 --version_no=version1  --model_type=${model_type} 

}

CUDA_VISIBLE_DEVICES=1 python trainShimaoka.py --is_test=False --datasets=Wiki --batch_size=2000 --class_size=128 --rnn_size=100 --type_dim=200 --attention_hidden_size=200 --sentence_length=62 --iterateEpoch=10 --epochs=5 --learning_rate=0.001 --threshold=0.5  --l2_loss_w=0.001 --margin=0.2 --max_neg_type=100 --max_pos_type=70 --max_pos_type_l1=10 --max_pos_type_l2=7  --restore=checkpoint/Wiki --init_epoch=0 --artifical_noise_weight=0.0 --iterate_num=0 --version_no=version1  --model_type=ShimaokeModel  --log_dir=logs/Wiki --is_add_fnode=False


CUDA_VISIBLE_DEVICES=1 python trainShimaoka.py --is_test=False --is_training=False --datasets=Wiki --batch_size=1000 --class_size=128 --rnn_size=100 --type_dim=500 --attention_hidden_size=200 --sentence_length=62 --iterateEpoch=50 --epochs=10 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.0 --margin=0.2 --max_neg_type=100 --max_pos_type=70 --max_pos_type_l1=10 --max_pos_type_l2=7  --restore=checkpoint/Wiki --init_epoch=0.0 --artifical_noise_weight=${artifical_noise_weight} --iterate_num=0 --version_no=version1  --model_type=DenoiseAbhishekModel  --log_dir=logs/Wiki/ --is_add_fnode=False --use_clean=True --ctx_length=15


CUDA_VISIBLE_DEVICES=0 python trainShimaoka.py --is_test=False --is_training=False --datasets=Wiki --batch_size=1000 --class_size=128 --rnn_size=100 --type_dim=500 --attention_hidden_size=200 --sentence_length=62 --iterateEpoch=50 --epochs=10 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.0 --margin=0.2 --max_neg_type=100 --max_pos_type=70 --max_pos_type_l1=10 --max_pos_type_l2=7  --restore=checkpoint/Wiki --init_epoch=${init_epoch} --artifical_noise_weight=${artifical_noise_weight} --iterate_num=0 --version_no=version1  --model_type=DenoiseAbhishekModel  --log_dir=logs/Wiki/ --is_add_fnode=False --use_clean=False --ctx_length=15

CUDA_VISIBLE_DEVICES=1 python trainShimaoka.py --is_training=False --is_test=True --datasets=Wiki --batch_size=2000 --class_size=128 --rnn_size=100 --type_dim=200 --attention_hidden_size=200 --sentence_length=62 --iterateEpoch=10 --epochs=30 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.0 --margin=0.2 --max_neg_type=100 --max_pos_type=70 --max_pos_type_l1=10 --max_pos_type_l2=7  --restore=checkpoint/Wiki/version1 --init_epoch=1 --artifical_noise_weight=0.0 --iterate_num=0 --version_no=version1  --model_type=PengModel  --log_dir=logs/Wiki/ --is_add_fnode=False  --alpha=0.4 

#for filter_threshold in 0.2
#do
#{
#  CUDA_VISIBLE_DEVICES=0 python gen_baseline_tag.py --is_test=False --is_training=False --datasets=Wiki --batch_size=2000 --class_size=128 --rnn_size=300 --type_dim=200 --attention_hidden_size=100 --sentence_length=62 --iterateEpoch=1 --epochs=20 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.001 --margin=0.2 --max_neg_type=100 --max_pos_type=70 --max_pos_type_l1=10 --max_pos_type_l2=7  --restore=checkpoint/Wiki/version1 --artifical_noise_weight=0.0 --iterate_num=0 --version_no=version1  --model_type=FullHierEnergyRandom  --filter_threshold=${filter_threshold} --init_epoch=0
#
#  for iterate_num in {6,7,8,9,10}
#  do
#  {
#    CUDA_VISIBLE_DEVICES=0 python trainReHierEnergy.py --is_test=False --datasets=Wiki --batch_size=2000 --class_size=128 --rnn_size=300 --type_dim=200 --attention_hidden_size=100 --sentence_length=62 --iterateEpoch=20 --epochs=1 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.001 --margin=0.2 --max_neg_type=100 --max_pos_type=70 --max_pos_type_l1=10 --max_pos_type_l2=7  --restore=checkpoint/Wiki/version1 --init_epoch=0 --artifical_noise_weight=0.0 --iterate_num=${iterate_num} --version_no=version1 --model_type=FullHierEnergyRandom --filter_threshold=${filter_threshold} --log_dir=logs/OntoNotes/Re_${filter_threshold}/iter_${iterate_num}/
#    CUDA_VISIBLE_DEVICES=0 python gen_iter_tag.py --is_test=False --is_training=False --datasets=Wiki --batch_size=2000 --class_size=128 --rnn_size=300 --type_dim=200 --attention_hidden_size=100 --sentence_length=62 --iterateEpoch=1 --epochs=20 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.001 --margin=0.2 --max_neg_type=100 --max_pos_type=70 --max_pos_type_l1=10 --max_pos_type_l2=7  --restore=checkpoint/Wiki/version1 --init_epoch=0 --artifical_noise_weight=0.0 --iterate_num=${iterate_num} --version_no=version1  --model_type=FullHierEnergyRandom --filter_threshold=${filter_threshold}
#  }
#  done
#}
#done
