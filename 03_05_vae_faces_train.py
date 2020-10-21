#!/usr/bin/env python
# coding: utf-8

# # 변이형 오토인코더 훈련 - 얼굴 데이터셋

# ## 라이브러리 임포트

# In[1]:


import os
from glob import glob
import numpy as np

from models.VAE import VariationalAutoencoder
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


# run params
section = 'vae'
run_id = '0001'
data_name = 'faces'
RUN_FOLDER = 'run/{}/'.format(section)
RUN_FOLDER += '_'.join([run_id, data_name])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' #'load' #


DATA_FOLDER = './data/celeb/'


# ## 데이터 적재

# *CelebA 사이트에서 `img_align_celeba.zip` 파일을 다운로드 받은 후 `data/celeb/` 폴더 안에서 압축을 해제하세요. `data/celeb/img_align_celeba` 폴더에 이미지가 저장되어야 합니다.*
# 
# *`list_attr_celeba.csv` 파일은 깃허브에 포함되어 있으므로 다운로드 받을 필요가 없습니다.*

# In[3]:


INPUT_DIM = (128,128,3)
BATCH_SIZE = 32

filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))

NUM_IMAGES = len(filenames)


# In[4]:


data_gen = ImageDataGenerator(rescale=1./255)

data_flow = data_gen.flow_from_directory(DATA_FOLDER
                                         , target_size = INPUT_DIM[:2]
                                         , batch_size = BATCH_SIZE
                                         , shuffle = True
                                         , class_mode = 'input'
                                         , subset = "training"
                                            )


# ## 모델 만들기

# In[5]:


vae = VariationalAutoencoder(
                input_dim = INPUT_DIM
                , encoder_conv_filters=[32,64,64, 64]
                , encoder_conv_kernel_size=[3,3,3,3]
                , encoder_conv_strides=[2,2,2,2]
                , decoder_conv_t_filters=[64,64,32,3]
                , decoder_conv_t_kernel_size=[3,3,3,3]
                , decoder_conv_t_strides=[2,2,2,2]
                , z_dim=200
                , use_batch_norm=True
                , use_dropout=True)

if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))


# In[6]:


vae.encoder.summary()


# In[7]:


vae.decoder.summary()


# ## 모델 훈련

# *주의: 이 훈련은 시간이 오래 걸립니다. 깃허브에 훈련된 모델이 포함되어 있으므로 아래 셀에서 VAE를 직접 훈련하지 않아도`03_06_vae_faces_analysis.ipynb` 노트북을 실행할 수 있습니다.*

# In[8]:


LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 10000
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0


# In[9]:


vae.compile(LEARNING_RATE, R_LOSS_FACTOR)


# In[10]:


vae.train_with_generator(     
    data_flow
    , epochs = EPOCHS
    , steps_per_epoch = NUM_IMAGES / BATCH_SIZE
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
)


# In[ ]:




