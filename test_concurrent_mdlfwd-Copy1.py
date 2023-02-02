# %%
from jp2tileaccesor.multi_res_Tiling import (
            SectionProxy, TileAccessor, Span, SectionMemmap, TileIterator
            )

import torch

from torch.utils.data import Dataset, DataLoader, Subset

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
from  torch.cuda.amp import autocast


##########

psnr = []
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



# %%

from collections import OrderedDict


class TileDataset(Dataset):
    def __init__(self,accessor):
        self.accessor = accessor
        
    def __len__(self):
        return len(self.accessor)
    
    def __getitem__(self,index):
        return torch.Tensor(self.accessor[index][0].copy())


import models.hovernet.net_desc as net
from run_utils.utils import convert_pytorch_checkpoint

def getmodel(model_path = 'hovernet_fast_pannuke_type_tf2pytorch.tar'):
    
    hovernet = net.HoVerNet(nr_types = 6,mode='fast')
    saved_state_dict = torch.load(model_path)["desc"]
    saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

    hovernet.load_state_dict(saved_state_dict, strict=True)
    
    return hovernet.eval()


def run_infer_load(dl,device):
    
    if device > 8:
        device = device%8
    dev = torch.device('cuda:'+str(device))
    
    torch.cuda.set_device(dev)
    
    start_load = time.perf_counter()
    
    mdl_gpu_half = getmodel().half().to(dev)
    mdl_gpu = getmodel().to(dev)
    
    start_i = time.perf_counter()
    
    _ = torch.set_grad_enabled(False)
    with autocast():
        for idx,batch in enumerate(dl):

            patch_imgs_gpu = batch.permute((0,3,1,2)).to(dev)
            fp32_patch = mdl_gpu(patch_imgs_gpu)
            patch_imgs_gpu = batch.permute((0,3,1,2)).half().to(dev)
            fp16_patch = mdl_gpu_half(patch_imgs_gpu)
            psnr.append()
    #         if idx%10==0:
    #             print(device,idx,time.perf_counter()-start_i)
            
    return device


def getdataset():
    proxy = SectionProxy()

    proxy.check_local_jp2()

    if not proxy.check_mmap():
        mmap = SectionMemmap(proxy)
        mmap.create()

    print(proxy.check_mmap())

# %%

    outsiz = 164,164
    insiz = 256,256
    hs = insiz[0]-outsiz[0]
    padsiz = hs//2
    accessor = TileAccessor(proxy,tilespan = Span(*outsiz),padding=padsiz, use_iip=False)

# %%

    ds = TileDataset(accessor)
    return ds


def main():
    
    ds = getdataset()
    print(len(ds))
    
    num_loaders = 8
    batch_siz = 128

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
            for v,d in executor.map(run_infer_load,loaders,range(num_loaders)):
                pass
    else:
        run_infer_load(loaders[0],0)
        
    end_time = time.perf_counter()
    print(end_time-start_time)

# %%

if __name__=="__main__":
    main()

