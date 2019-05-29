TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_INC='/home/wjs/anaconda2/envs/tensorflow/lib/python2.7/site-packages/tensorflow/include'
CUDA_PATH='/usr/local/cuda'
echo $CUDA_PATH
echo $TF_INC
for i in *.cc
do
{
  echo $i
  g++ -std=c++11 -shared $i -o ${i::-2}so -I $TF_INC -fPIC
}
done