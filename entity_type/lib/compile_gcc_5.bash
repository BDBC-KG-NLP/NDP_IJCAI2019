TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
echo ${TF_LIB}
for i in *.cc; do
  echo $i
  g++ -std=c++11 -shared $i -o ${i::-2}so -fPIC -I $TF_INC -D_GLIBCXX_USE_CXX11_ABI=0 -L$TF_LIB -ltensorflow_framework
done
