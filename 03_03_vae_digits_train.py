#!/usr/bin/env python
# coding: utf-8

# # 변이형 오토인코더 훈련

# ## 라이브러리 임포트

# In[1]:


import os

from models.VAE import VariationalAutoencoder
from utils.loaders import load_mnist


# In[2]:


# 실행 매개변수
SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' #'load' #


# ## 데이터 적재

# In[3]:


(x_train, y_train), (x_test, y_test) = load_mnist()


# ## 모델 만들기

# In[4]:


vae = VariationalAutoencoder(
    input_dim = (28,28,1)
    , encoder_conv_filters = [32,64,64, 64]
    , encoder_conv_kernel_size = [3,3,3,3]
    , encoder_conv_strides = [1,2,2,1]
    , decoder_conv_t_filters = [64,64,32,1]
    , decoder_conv_t_kernel_size = [3,3,3,3]
    , decoder_conv_t_strides = [1,2,2,1]
    , z_dim = 2
)

if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))


# In[5]:


vae.encoder.summary()


# In[6]:


from keras.utils import plot_model
plot_model(vae.encoder, to_file='vae_encoder.png', show_shapes = True, show_layer_names = True)


# ![VAE Encoder](vae_encoder.png "VAE Encoder")

# In[7]:


vae.decoder.summary()


# ## 모델 훈련

# In[8]:


LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 1000


# In[9]:


vae.compile(LEARNING_RATE, R_LOSS_FACTOR)


# In[10]:


BATCH_SIZE = 32
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0


# In[11]:


vae.train(     
    x_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
)

