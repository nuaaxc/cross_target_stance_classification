This is a keras implementation of CrossNet for paper Cross-Target Stance Classification with Self-Attention Networks (https://arxiv.org/abs/1805.06593)

The experimental process consists of three steps:

1) preprocessing (data_util.py and tokenizer): convert tweet text, target phrase, and label into internal matrix format, with shapes (batch_size, sent_length), (batch_size, target_length), (batch_size, num_class)

2) training and testing (train_model_CrossNet_keras.py)
  config.py: all directory and model configurations are here
  models/CrossNet.py: model implementation of CrossNet
  models/layers.py: layer implementation of CrossNet

Requirements:
  python 3.6
  keras 2.1.3
  tensorflow 1.6

Usage:
On windows (Train and test):
set PYTHONPATH=%PYTHONPATH%;C:\path_to_project\cross_target_stance_classification\
C:\path_to_python\python.exe C:\path_to_project\cross_target_stance_classification\train_model_CrossNet_keras.py -tr_te --target cc_cc --n_aspect 1 --bsize 128 --rnn_dim 128 --dense_dim 64 --dropout_rate 0.2 --max_epoch 200 --learning_rate 0.001
