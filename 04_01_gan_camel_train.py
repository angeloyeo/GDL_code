#!/usr/bin/env python
# coding: utf-8

# # GAN 훈련

# *Note: 최신 버전의 라이브러리를 사용하기 때문에 책의 내용과 결과가 다를 수 있습니다*

# ## 라이브러리 임포트

# In[1]:


import os
import matplotlib.pyplot as plt

from models.GAN import GAN
from utils.loaders import load_safari


# In[2]:


# run params
SECTION = 'gan'
RUN_ID = '0001'
DATA_NAME = 'camel'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' #'load' #


# ## 데이터 적재

# *깃허브에 camel 데이터셋이 포함되어 있으므로 별도로 다운로드 받을 필요가 없습니다.*

# In[4]:


(x_train, y_train) = load_safari(DATA_NAME)


# In[5]:


x_train.shape


# In[6]:


plt.imshow(x_train[200,:,:,0], cmap = 'gray')


# ## 모델 만들기

# In[7]:


gan = GAN(input_dim = (28,28,1)
        , discriminator_conv_filters = [64,64,128,128]
        , discriminator_conv_kernel_size = [5,5,5,5]
        , discriminator_conv_strides = [2,2,2,1]
        , discriminator_batch_norm_momentum = None
        , discriminator_activation = 'relu'
        , discriminator_dropout_rate = 0.4
        , discriminator_learning_rate = 0.0008
        , generator_initial_dense_layer_size = (7, 7, 64)
        , generator_upsample = [2,2, 1, 1]
        , generator_conv_filters = [128,64, 64,1]
        , generator_conv_kernel_size = [5,5,5,5]
        , generator_conv_strides = [1,1, 1, 1]
        , generator_batch_norm_momentum = 0.9
        , generator_activation = 'relu'
        , generator_dropout_rate = None
        , generator_learning_rate = 0.0004
        , optimiser = 'rmsprop'
        , z_dim = 100
        )

if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))


# In[8]:


gan.discriminator.summary()


# In[9]:


gan.generator.summary()


# ## 모델 훈련

# *에포크마다 생성된 샘플 이미지가 `run/gan/0001_camel/images` 폴더에 저장됩니다.*

# In[10]:


BATCH_SIZE = 64
EPOCHS = 6000
PRINT_EVERY_N_BATCHES = 5


# In[11]:


gan.train(     
    x_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
)


# In[12]:


fig = plt.figure()
plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

plt.plot([x[1] for x in gan.d_losses], color='green', linewidth=0.25)
plt.plot([x[2] for x in gan.d_losses], color='red', linewidth=0.25)
plt.plot([x[0] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.xlim(0, 2000)
plt.ylim(0, 2)

plt.show()


# In[13]:


fig = plt.figure()
plt.plot([x[3] for x in gan.d_losses], color='black', linewidth=0.25)
plt.plot([x[4] for x in gan.d_losses], color='green', linewidth=0.25)
plt.plot([x[5] for x in gan.d_losses], color='red', linewidth=0.25)
plt.plot([x[1] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('accuracy', fontsize=16)

plt.xlim(0, 2000)

plt.show()


# #### 에포크 20
# ![20](run/gan/0001_camel/images/sample_20.png)

# #### 에포크 200
# ![200](run/gan/0001_camel/images/sample_200.png)

# #### 에포크 400
# ![400](run/gan/0001_camel/images/sample_400.png)

# #### 에포크 1000
# ![1000](run/gan/0001_camel/images/sample_1000.png)

# #### 에포크 2000
# ![2000](run/gan/0001_camel/images/sample_2000.png)
