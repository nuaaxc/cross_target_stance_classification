This is a keras implementation of CrossNet for cross-target stance detection

The process consists of three steps:

1) preprocessing (data_util.py and tokenizer): convert raw text into internal format - a matrix (batch_size,  sequence_length)

2) training (train_model_CrossNet_keras.py)

3) testing and evaluation (train_model_CrossNet_keras.py)

config.py: contains all directory and model configurations
models/CrossNet.py: implementation of CrossNet
models/layers.py: layer implementation of CrossNet

Requirements:
  keras 2.0.8
  tensorflow 1.3
