TF_INC=$(python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') 
echo ${TF_INC}
for i in *.cc
do
{
echo $i
}
done
