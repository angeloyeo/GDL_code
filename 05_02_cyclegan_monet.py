#!/usr/bin/env python
# coding: utf-8

# # CycleGAN 모네 그림 그리기

# ## 라이브러리 임포트

# *Note: 이 노트북의 코드를 실행하려면 `keras_contrib` 패키지를 설치해야 합니다. 다음 셀의 주석을 제거하고 실행하여 패키지를 설치하세요*

# In[2]:


#!pip install git+https://www.github.com/keras-team/keras-contrib.git


# In[3]:


import os
import matplotlib.pyplot as plt

from models.cycleGAN import CycleGAN
from utils.loaders import DataLoader


# In[4]:


# run params
SECTION = 'paint'
RUN_ID = '0001'
DATA_NAME = 'monet2photo'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' # 'build' # 


# ## 데이터 적재

# 노트북을 처음 실행할 때 다음 셀의 주석을 제거하고 실행하여 모네 데이터셋을 다운로드하세요.

# In[ ]:


#!./scripts/download_cyclegan_data.sh monet2photo


# In[8]:


IMAGE_SIZE = 256


# In[9]:


data_loader = DataLoader(dataset_name=DATA_NAME, img_res=(IMAGE_SIZE, IMAGE_SIZE))


# ## 모델 생성

# In[10]:


gan = CycleGAN(
        input_dim = (IMAGE_SIZE,IMAGE_SIZE,3)
        , learning_rate = 0.0002
        , lambda_validation = 1
        , lambda_reconstr = 10
        , lambda_id = 5
        , generator_type = 'resnet'
        , gen_n_filters = 32
        , disc_n_filters = 64
        )

if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))


# In[11]:


gan.g_BA.summary()


# In[12]:


gan.g_AB.summary()


# In[13]:


gan.d_A.summary()


# In[14]:


gan.d_B.summary()


# ## 모델 훈련

# In[17]:


BATCH_SIZE = 1
EPOCHS = 3
PRINT_EVERY_N_BATCHES = 10

TEST_A_FILE = '00490.jpg'
TEST_B_FILE = '2014-08-02 15:56:41.jpg'


# In[18]:


gan.train(data_loader
        , run_folder = RUN_FOLDER
        , epochs=EPOCHS
        , test_A_file = TEST_A_FILE
        , test_B_file = TEST_B_FILE
        , batch_size=BATCH_SIZE
        , sample_interval=PRINT_EVERY_N_BATCHES)


# ## 훈련 결과

# 1 배치 후
# ![1배치](run/paint/0001_monet2photo/images/0_0_0.png)
# 
# 1 에포크
# ![1에포크](run/paint/0001_monet2photo/images/0_0_990.png)
# 
# 2 에포크
# ![1에포크](run/paint/0001_monet2photo/images/0_1_990.png)
# 
# 3 에포크
# ![1에포크](run/paint/0001_monet2photo/images/0_2_990.png)
