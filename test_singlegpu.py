from jp2tileaccesor.multi_res_Tiling import (
            SectionProxy, TileAccessor, Span, SectionMemmap, TileIterator
            )

import torch

from torch.utils.data import Dataset, DataLoader, Subset

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import sys


##########
def trt_version():
    return trt.__version__


def torch_version():
    return torch.__version__

def torch_dtype_to_trt(dtype):
    if trt_version() >= '7.0' and dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError("%s is not supported by tensorrt" % dtype)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device("cuda").type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device("cpu").type:
        return trt.TensorLocation.HOST
    else:
        return TypeError("%s is not supported by tensorrt" % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)

import tensorrt as trt
TRT_LOGGER = trt.Logger(min_severity =trt.ILogger.INTERNAL_ERROR)



#%%

from collections import OrderedDict


class TileDataset(Dataset):
    def __init__(self,accessor,num_loaders):
        self.accessor = accessor
        self.TensorDataset = {}
        self.num_loaders = num_loaders
        self.preload()
    
    def preload(self):
        print("preloading")
        device_id = 0
        flag = True
        dev = torch.device('cuda:'+str(device_id))
        print(len(self.accessor))
        chunk = len(self.accessor)//self.num_loaders
        print(chunk)
#         print(sys.getsizeof(self.accessor[0][0])*len(self.accessor)/(1024*1024*1024))
        for i in range(len(self.accessor)):
            if(i%chunk == 0):
                if(flag == True):
                    flag = False
                    pass
                else:
                    device_id+=1
                    dev = torch.device('cuda:'+str(device_id))
            self.TensorDataset[i] = torch.Tensor(self.accessor[i][0].copy()).to(dev)
        
    def __len__(self):
        return len(self.accessor)
    
    def __getitem__(self,index):
        return self.TensorDataset[index]
    

# def preload(dl,dev):
#     start_load = time.perf_counter()
#     gpu_ptrs = {}
#     for idx,batch in enumerate(dl):
#         patch_imgs_gpu = batch.permute((0,3,1,2)).float().to(dev)
#         patch_imgs_gpu = patch_imgs_gpu.contiguous()
#         gpu_ptrs[idx] = patch_imgs_gpu.data_ptr()
#     print("preloading done : ",time.perf_counter()-start_load, dev)
#     return gpu_ptrs
    
def run_infer_load_preload(dl,device):
    if device > 8:
        device = device%8
    dev = torch.device('cuda:'+str(device))
    torch.cuda.set_device(dev)
    
    start_load = time.perf_counter()
    out = []
    with open("./hbp_hover_net/hovernet_256_64_best.plan", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    print(f'model load({device}):',time.perf_counter()-start_load)
    
    start_i = time.perf_counter()
    
    for idx,batch in enumerate(dl):
        
        d_input = batch.data_ptr()
        
        pred_dict = OrderedDict()
        pred_tensor = torch.empty(
            size = (64,10,164,164) , 
            dtype = torch.float32, 
                device = dev)
        
        pred_tensor = pred_tensor.to(dev)
        d_output = pred_tensor.data_ptr()

        with engine.create_execution_context() as context:
            context.execute_v2(bindings=[int(d_input), int(d_output)])

        out_i = pred_tensor.reshape((64,10, 164, 164))   
        out.append(out_i.cpu())
        if idx%10==0:
            print(device,idx,time.perf_counter()-start_i)
        
    return out,device

    
def run_infer_load(dl,device):
    if device > 8:
        device = device%8
    dev = torch.device('cuda:'+str(device))
    
    torch.cuda.set_device(dev)
    
    start_load = time.perf_counter()
    with open("./hbp_hover_net/hovernet_256_64_best.plan", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    print(f'model load({device}):',time.perf_counter()-start_load)
    
    out = []
    start_i = time.perf_counter()
    for idx,batch in enumerate(dl):
        
        patch_imgs_gpu = batch.permute((0,3,1,2)).float().to(dev)
        patch_imgs_gpu = patch_imgs_gpu.contiguous()
        d_input = patch_imgs_gpu.data_ptr()
        
        pred_dict = OrderedDict()
        pred_tensor = torch.empty(
            size = (batch.shape[0],10,164,164) , 
            dtype = torch.float32, 
                device = dev)
        
        pred_tensor = pred_tensor.to(dev)
        d_output = pred_tensor.data_ptr()

        with engine.create_execution_context() as context:
            context.execute_v2(bindings=[int(d_input), int(d_output)])

        out_i = pred_tensor.reshape((batch.shape[0],10, 164, 164))
        out.append(out_i.cpu())
        if idx%10==0:
            print(device,idx,time.perf_counter()-start_i)
        
    return out,device


def main():
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
    num_loaders = 8
    batch_siz = 64
#     dev = torch.device('cuda:'+str(0))
    
    start_load = time.perf_counter()
    ds = TileDataset(accessor,num_loaders)
    print(time.perf_counter() - start_load)
    print(len(ds))


    partsiz = len(ds)//num_loaders
    # print(partsiz)

    loaders = []
    for ii in range(num_loaders):
        # print(ii*partsiz,(ii+1)*partsiz)
        part = range(ii*partsiz,(ii+1)*partsiz)
        ds_sub = Subset(ds, part)

        loaders.append(DataLoader(ds_sub, batch_size = batch_siz, num_workers=12, drop_last=True))

    print('start')
    # concurrent
    start_time = time.perf_counter()
    if num_loaders > 1:
        with ThreadPoolExecutor(max_workers=num_loaders) as executor:
            for v,d in executor.map(run_infer_load_preload,loaders,range(num_loaders)):
                pass
    else:
        run_infer_load_preload(loaders[0],0)
        
    end_time = time.perf_counter()
    print(end_time-start_time)

    #%%

if __name__=="__main__":
    main()
    