#!/usr/bin/env python
# coding: utf-8

# # CycleGAN 훈련

# *Note: 라이브러리 버전 때문에 책의 내용과 결과가 다를 수 있습니다*

# ## 라이브러리 임포트

# *Note: 이 노트북의 코드를 실행하려면 `keras_contrib` 패키지를 설치해야 합니다. 다음 셀의 주석을 제거하고 실행하여 패키지를 설치하세요*

# In[1]:


#!pip install git+https://www.github.com/keras-team/keras-contrib.git


# In[2]:


import os
import matplotlib.pyplot as plt

from models.cycleGAN import CycleGAN
from utils.loaders import DataLoader


# In[3]:


# run params
SECTION = 'paint'
RUN_ID = '0001'
DATA_NAME = 'apple2orange'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' # 'build' # 


# ## 데이터 적재

# 노트북을 처음 실행할 때 다음 셀의 주석을 제거하고 실행하여 사과, 오렌지 데이터셋을 다운로드하세요.

# In[4]:


#!./scripts/download_cyclegan_data.sh apple2orange


# In[5]:


IMAGE_SIZE = 128


# In[6]:


data_loader = DataLoader(dataset_name=DATA_NAME, img_res=(IMAGE_SIZE, IMAGE_SIZE))


# ## 모델 생성

# In[7]:


gan = CycleGAN(
    input_dim = (IMAGE_SIZE,IMAGE_SIZE,3)
    ,learning_rate = 0.0002
    , buffer_max_length = 50
    , lambda_validation = 1
    , lambda_reconstr = 10
    , lambda_id = 2
    , generator_type = 'unet'
    , gen_n_filters = 32
    , disc_n_filters = 32
    )

if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))


# In[8]:


gan.g_BA.summary()


# In[9]:


gan.g_AB.summary()


# In[10]:


gan.d_A.summary()


# In[11]:


gan.d_B.summary()


# ## 모델 훈련

# *Note: CycleGAN 훈련 도중 주피터 커널이 예기치 않게 종료될 수 있습니다. 이럴 때는 쉘에서 05_01_cyclegan_train.py 파일을 실행하여 훈련하세요.*

# In[12]:


BATCH_SIZE = 1
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 10

TEST_A_FILE = 'n07740461_14740.jpg'
TEST_B_FILE = 'n07749192_4241.jpg'


# *Note: 이 훈련은 시간이 매우 오래 걸립니다. 깃허브에 훈련된 가중치와 손실이 저장되어 있으므로 훈련을 건너 뛰고 다음 셀을 실행해도 됩니다.*

# In[ ]:


gan.train(data_loader
        , run_folder = RUN_FOLDER
        , epochs=EPOCHS
        , test_A_file = TEST_A_FILE
        , test_B_file = TEST_B_FILE
        , batch_size=BATCH_SIZE
        , sample_interval=PRINT_EVERY_N_BATCHES)


# ## 결과

# ![apple2orange](run/paint/0001_apple2orange/images/0_199_990.png)

# ## 손실

# *Note: 앞에서 훈련을 직접 실행하지 않았다면 다음 셀의 주석을 제거하고 실행하세요*

# In[ ]:


# !gunzip run/paint/0001_apple2orange/loss.pkl.gz


# In[13]:


# import pickle

# loss = pickle.load(open(os.path.join(RUN_FOLDER, "loss.pkl"), "rb"))

# gan.d_losses = loss['d_losses']
# gan.g_losses = loss['g_losses']


# In[14]:


fig = plt.figure(figsize=(20,10))

# plt.plot([x[0] for x in gan.g_losses], color='black', linewidth=0.25)
plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.1) #discriminator loss

plt.plot([x[1] for x in gan.g_losses], color='green', linewidth=0.1) #validation loss
# plt.plot([x[2] for x in gan.g_losses], color='orange', linewidth=0.1)

plt.plot([x[3] for x in gan.g_losses], color='blue', linewidth=0.1) #reconstr loss
# plt.plot([x[4] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.plot([x[5] for x in gan.g_losses], color='red', linewidth=0.1) #id loss
# plt.plot([x[6] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.ylim(0, 5)

plt.show()

