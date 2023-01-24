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


from jp2tileaccesor.multi_res_Tiling import (
            SectionProxy, TileAccessor, Span, SectionMemmap, TileIterator
            )

proxy = SectionProxy()

proxy.check_local_jp2()

if not proxy.check_mmap():
    mmap = SectionMemmap(proxy)
    mmap.create()

print(proxy.check_mmap())

#%%

outsiz = 164,164
insiz = 256,256
hs = insiz[0]-outsiz[0]
padsiz = hs//2
accessor = TileAccessor(proxy,tilespan = Span(*outsiz),padding=padsiz, use_iip=False)

#%%

iterator = TileIterator(accessor)

# arr,rgn,url = next(iterator)

# print(arr.shape)




# In[3]:


len(accessor)


# In[4]:


total = 31747*8


# In[5]:


num_gpu = 8
chunk = total//num_gpu


# In[9]:


chunk


# In[10]:


brain_batches = [i for i in range(0,total,chunk)]


# In[11]:


222229+ chunk


# In[12]:


brain_batches


# ## Multi-GPU inference on TensorRT using ThreadPool

# In[13]:


# get_ipython().system('ls | grep .plan')


# In[14]:


from concurrent.futures import ThreadPoolExecutor
import TensorRT.InferMultiGPU as gpu
print(os.cpu_count())


# In[15]:


model_name = "hovernet_256_16_best.plan"


# In[16]:


batch_size = 16
Image_size = 256


# In[17]:


# infer_gpu = []
# infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,0))
# infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,1))


# In[18]:


import numpy as np


# In[19]:


def make_batch(start,end):
    np_pics = accessor[start][0]
    brain_tiles = np_pics.reshape(1,3,Image_size,Image_size)
    for i in range(start+1,end):
        img = accessor[i][0]
        
        if(img.shape == (256,256,3)):
            brain_tiles = np.concatenate((brain_tiles,img.reshape(1,3,Image_size,Image_size)),axis=0)
        else:
            w_diff = 256-img.shape[0]
            h_diff = 256-img.shape[1]
            img = np.pad(img,[(w_diff,0),(h_diff,0),(0,0)],mode='reflect')
            brain_tiles = np.concatenate((brain_tiles,img.reshape(1,3,Image_size,Image_size)),axis=0)
    return brain_tiles
    
def iterate_batch(gpu,batch_data):
    for i in range(batch_data,batch_data + chunk,batch_size):
        out = gpu.run_inf(make_batch(i,i+batch_size))
    gpu.clean()
    return out


# In[ ]:


# %time
# infer_gpu = []
# infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,0))

# for i in range(brain_batches[0],brain_batches[0] + chunk,batch_size):
#     out = infer_gpu[0].run_inf(make_batch(i,i+batch_size))


# In[ ]:


infer_gpu = []
infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,0))
infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,1))
infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,2))
infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,3))
infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,4))
infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,5))
infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,6))
infer_gpu.append(gpu.GPU_infer(model_name,batch_size,trt.float32,7))
with ThreadPoolExecutor(8) as executor:
    results = executor.map(iterate_batch,infer_gpu,brain_batches)


# In[ ]:





# In[16]:


# %%timeit
# infer_gpu = []
# infer_gpu.append(gpu.GPU_infer(model_name,8,trt.float32,0))
# infer_gpu.append(gpu.GPU_infer(model_name,8,trt.float32,1))
# with ThreadPoolExecutor(2) as executor:
#     results = executor.map(iterate_batch,infer_gpu,brain_batches)


# In[ ]:


# for r in results:
#     print(r)


# In[17]:





# In[26]:


# infer_gpu[0].h_output.dtype,type(infer_gpu[0].h_output)


# In[18]:


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


# In[24]:


visualize_channel(out,3)


# ## Multi-GPU inference on Torch using ThreadPool

# In[12]:


# import os
# from PIL import Image
# import numpy as np
# import tensorrt
# import onnx
# import gc
# import models.hovernet.net_desc as net
# import torch
# import matplotlib.pyplot as plt
# torch.set_grad_enabled(False)
# Image_size = 256


# In[1]:


# from run_utils.utils import convert_pytorch_checkpoint

# model_path = 'hovernet_fast_pannuke_type_tf2pytorch.tar'
# hovernet = net.HoVerNet(nr_types = 6,mode='fast')
# saved_state_dict = torch.load(model_path)["desc"]
# saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

# hovernet.load_state_dict(saved_state_dict, strict=True)
# hovernet = torch.nn.parallel.DataParallel(hovernet,device_ids = [0,1])
# hovernet = hovernet.to('cuda')
# hovernet.eval()


# In[23]:


# %%timeit
# for i in range(0,brain_tiles.shape[0],4):
#     out = hovernet.forward(brain_tiles[i:i+4,:,:,:])


# In[14]:


# %%timeit
# for i in range(0,brain_tiles.shape[0],8):
#     out = hovernet.forward(brain_tiles[i:i+8,:,:,:])


# In[ ]:





# In[ ]:




