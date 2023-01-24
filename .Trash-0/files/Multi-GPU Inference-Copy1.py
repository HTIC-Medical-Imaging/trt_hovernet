#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
import tensorrt as trt
import torch
import matplotlib.pyplot as plt


# In[2]:


Image_size = 256
storage_dir = "./storage"
file_list = os.listdir(storage_dir+"/cache_"+str(Image_size))
print("Number of Images",len(file_list))
pics = [ storage_dir + "/cache_"+str(Image_size) + "/" +str(i) for i in file_list]


# In[3]:


np_pics = np.asarray(Image.open(pics[0]))
brain_tiles = np_pics.reshape(1,3,Image_size,Image_size)
for i in range(257):
    img = np.asarray(Image.open(pics[i+1]))
    if (img.shape == (256, 256, 3)):
        brain_tiles = np.concatenate((brain_tiles,img.reshape(1,3,Image_size,Image_size)),axis=0)
    


# In[4]:


brain_tiles = torch.from_numpy(brain_tiles).float()


# In[5]:


brain_tiles = brain_tiles[:1024,:3,:,:]


# In[6]:


brain_tiles.shape


# In[7]:


num_gpu = 2
chunk = brain_tiles.shape[0]//num_gpu


# In[8]:


brain_batches = [brain_tiles[i:i+chunk,:,:,:] for i in range(0,brain_tiles.shape[0],chunk)]


# In[9]:


len(brain_batches)


# In[10]:


brain_batches[0].shape


# In[11]:


brain_batches[1].shape


# In[12]:


from concurrent.futures import ThreadPoolExecutor
import TensorRT.InferMultiGPU as gpu
print(os.cpu_count())


# In[13]:


model_name = "hovernet_256_4_fp16.plan"


# In[14]:


infer_gpu = []


# In[15]:


infer_gpu.append(gpu.GPU_infer(model_name,4,trt.float32,0))
infer_gpu.append(gpu.GPU_infer(model_name,4,trt.float32,1))


# In[16]:


def iterate_batch(gpu,images):
    for i in range(0,images.shape[0],4):
        gpu.run_inf(images[i:i+4,:,:,:])
    gpu.clean()


# In[ ]:


with ThreadPoolExecutor(2) as executor:
    results = executor.map(iterate_batch,infer_gpu,brain_batches)


# In[18]:


for r in results:
    print(r)


# In[18]:


# iterate_batch(infer_gpu[0],brain_batches[0])


# In[19]:


# def visualize_channel(out,img_no = 0):
#     fig , axs = plt.subplots(5,2,figsize=(12,15))
#     k=0
#     plt.axis('off')
#     count = 0
#     for i in range(5):
#         for j in range(2):
#             axs[i,j].xaxis.set_visible(False)
#             axs[i,j].yaxis.set_visible(False)
#             t = axs[i,j].imshow(out[img_no,k,:,:])
#             plt.colorbar(t)
#             axs[i,j].title.set_text( str(k) + " channel ")
#             k+=1


# In[20]:


# visualize_channel(out,2)


# In[ ]:




