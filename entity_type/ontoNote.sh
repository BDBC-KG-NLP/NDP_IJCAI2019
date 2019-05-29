#insert data into mongoDB
CUDA_VISIBLE_DEVICES=0 python insert_knet_2_mongo.py --is_test=False --datasets=OntoNotes --batch_size=1000 --class_size=89 --rnn_size=200 --attention_hidden_size=100 --type_dim=200 --sentence_length=250


for model_type in FullHierEnergy FullHierEnergyContext FullHierEnergyRandom
do
{
  CUDA_VISIBLE_DEVICES=1 python trainHierEnergy.py --is_test=False --datasets=OntoNotes --batch_size=1000 --class_size=89 --rnn_size=200 --attention_hidden_size=100 --type_dim=200 --sentence_length=250 --iterateEpoch=30 --epochs=4 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.01 --margin=0.4 --max_neg_type=70 --max_pos_type=180 --max_pos_type_l1=6 --max_pos_type_l2=6 --max_pos_type_l3=5 --init_epoch=0  --artifical_noise_weight=0.0 --restore=checkpoint/OntoNotes/version1 --version_no=version1 --model_type=${model_type} --is_add_fnode=False --log_dir=logs/OntoNotes/0.001/

}

#you can also  tune the parameter
for model_type in ComplExModel,ComplExContextModel,ComplExRandomModel
do
{
  CUDA_VISIBLE_DEVICES=0 python trainShimaoka.py --is_test=False --datasets=OntoNotes --batch_size=1000 --class_size=89 --rnn_size=200 --attention_hidden_size=100 --type_dim=200 --sentence_length=250 --iterateEpoch=2 --epochs=4 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.0001 --margin=0.4 --max_neg_type=70 --max_pos_type=180 --max_pos_type_l1=6 --max_pos_type_l2=6 --max_pos_type_l3=5 --init_epoch=0 --artifical_noise_weight=0.0 --restore=checkpoint/OntoNotes/version2 --version_no=version2 --model_type=${model_type} --is_add_fnode=False --log_dir=logs/OntoNotes/1.0/
}

CUDA_VISIBLE_DEVICES=0 python trainShimaoka.py --datasets=OntoNotes --batch_size=1000 --class_size=89 --rnn_size=100 --attention_hidden_size=20 --type_dim=100 --sentence_length=250 --iterateEpoch=1 --epochs=5 --learning_rate=0.001 --threshold=0.5 --l2_loss_w=0.005 --margin=0.4 --max_neg_type=30 --max_pos_type=11 --init_epoch 10 --is_test=False --artifical_noise_weight=0.0 --version_no=version1 --model_type=ShimaokeModel


CUDA_VISIBLE_DEVICES=0 python trainShimaoka.py --is_test=True --datasets=OntoNotes --batch_size=1000 --class_size=89 --rnn_size=200 --attention_hidden_size=100 --type_dim=200 --sentence_length=250 --iterateEpoch=10 --epochs=10 --learning_rate=0.001 --threshold=0.5  --l2_loss_w=0.0 --margin=0.4 --max_neg_type=70 --max_pos_type=12 --restore=checkpoint/OntoNotes/version1 --version_no=version1 --init_epoch=0  --artifical_noise_weight=0.0 --model_type=ShimaokeModel

CUDA_VISIBLE_DEVICES=1 python trainShimaoka.py --is_test=True --datasets=OntoNotes --batch_size=1000  --class_size=89 --rnn_size=200 --attention_hidden_size=100 --type_dim=200 --sentence_length=250 --iterateEpoch=10 --epochs=10 --learning_rate=0.01 --threshold=0.0  --l2_loss_w=0.0 --margin=0.4 --max_neg_type=70 --max_pos_type=12 --restore=checkpoint/OntoNotes/version1  --init_epoch=0  --artifical_noise_weight=0.0 --version_no=version1 --model_type=DenoiseAbhishekModel --use_clean=False

CUDA_VISIBLE_DEVICES=1 python trainShimaoka.py --is_test=True --datasets=OntoNotes --batch_size=1000  --class_size=89 --rnn_size=200 --attention_hidden_size=100 --type_dim=200 --sentence_length=250 --iterateEpoch=10 --epochs=10 --learning_rate=0.01 --threshold=0.0   --l2_loss_w=0.0 --margin=0.4 --max_neg_type=70 --max_pos_type=12 --restore=checkpoint/OntoNotes/version1  --init_epoch=0  --artifical_noise_weight=0.0 --version_no=version1 --model_type=DenoiseAbhishekModel --use_clean=True

CUDA_VISIBLE_DEVICES=0 python trainShimaoka.py --is_test=True --datasets=OntoNotes --batch_size=1000  --class_size=89 --rnn_size=200 --attention_hidden_size=100 --type_dim=200 --sentence_length=250 --iterateEpoch=10 --epochs=10 --learning_rate=0.001 --threshold=0.0   --l2_loss_w=0.0001 --margin=0.4 --max_neg_type=70 --max_pos_type=12 --restore=checkpoint/OntoNotes/version1  --init_epoch=0 --artifical_noise_weight=0.0 --version_no=version1 --model_type=PengModel --use_clean=False --alpha=0.3


##we also provide iterative training for OntoNotes
#for filter_threshold in 0.0
#do
#{
#   CUDA_VISIBLE_DEVICES=1 python gen_baseline_tag.py --is_test=False --datasets=OntoNotes --batch_size=1000 --class_size=89 --rnn_size=100 --attention_hidden_size=100 --type_dim=200 --sentence_length=250 --iterateEpoch=2 --epochs=10 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.01 --margin=0.4 --max_neg_type=70 --max_pos_type=180 --max_pos_type_l1=6 --max_pos_type_l2=6 --max_pos_type_l3=5 --init_epoch=0  --artifical_noise_weight=0.0 --restore=checkpoint/OntoNotes/version3 --version_no=version3 --model_type=FullHierEnergyRandom --is_add_fnode=True --filter_threshold=${filter_threshold} --iterate_num=0
#  for iterate_num in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
#  do
#  {
#  CUDA_VISIBLE_DEVICES=1 python trainReHierEnergy.py --is_test=False --datasets=OntoNotes --batch_size=1000 --class_size=89 --rnn_size=100 --attention_hidden_size=100 --type_dim=200 --sentence_length=250 --iterateEpoch=2 --epochs=1 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.01 --margin=0.4 --max_neg_type=70 --max_pos_type=180 --max_pos_type_l1=6 --max_pos_type_l2=6 --max_pos_type_l3=5 --init_epoch=0  --artifical_noise_weight=0.0 --restore=checkpoint/OntoNotes/version3 --version_no=version3 --model_type=FullHierEnergyRandom --is_add_fnode=True --filter_threshold=${filter_threshold} --iterate_num=${iterate_num}
#  
#  CUDA_VISIBLE_DEVICES=1 python gen_iter_tag.py --is_test=False --datasets=OntoNotes --batch_size=1000 --class_size=89 --rnn_size=100 --attention_hidden_size=100 --type_dim=200 --sentence_length=250 --iterateEpoch=2 --epochs=10 --learning_rate=0.001 --threshold=0.0  --l2_loss_w=0.01 --margin=0.4 --max_neg_type=70 --max_pos_type=180 --max_pos_type_l1=6 --max_pos_type_l2=6 --max_pos_type_l3=5 --init_epoch=0  --artifical_noise_weight=0.0 --restore=checkpoint/OntoNotes/version3 --version_no=version3 --model_type=FullHierEnergyRandom --is_add_fnode=True --filter_threshold=${filter_threshold} --iterate_num=${iterate_num}
#  }
#  done
#}
#done