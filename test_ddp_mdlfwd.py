import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


import models.hovernet.net_desc as net
from run_utils.utils import convert_pytorch_checkpoint

def getmodel(model_path = 'hovernet_fast_pannuke_type_tf2pytorch.tar'):
    
    hovernet = net.HoVerNet(nr_types = 6,mode='fast')
    saved_state_dict = torch.load(model_path)["desc"]
    saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

    hovernet.load_state_dict(saved_state_dict, strict=True)
    return hovernet


from jp2tileaccesor.multi_res_Tiling import (
        SectionProxy, TileAccessor, Span, SectionMemmap, TileIterator
    )
from torch.utils.data import Dataset, DataLoader

class TileDataset(Dataset):
    def __init__(self,accessor):
        self.accessor = accessor
        
    def __len__(self):
        return len(self.accessor)
    
    def __getitem__(self,index):
        return torch.Tensor(self.accessor[index][0].copy())
    
def getdataset():
    proxy = SectionProxy()

    proxy.check_local_jp2()

    if not proxy.check_mmap():
        mmap = SectionMemmap(proxy)
        mmap.create()

    print(proxy.check_mmap())

    outsiz = 164,164
    insiz = 256,256
    hs = insiz[0]-outsiz[0]
    padsiz = hs//2
    accessor = TileAccessor(proxy,tilespan = Span(*outsiz),padding=padsiz, use_iip=False)

    ds = TileDataset(accessor)
    return ds

def run_infer_load(rank, world_size, model, dl):
    # create default process group
    dist.init_process_group("nccl",init_method='tcp://localhost:23456', rank=rank, world_size=world_size)
    
    # construct DDP model
    ddp_model = DDP(model.to(rank), device_ids=[rank])
    
    # forward pass
    outputs = []
    for idx,batch in enumerate(dl):
        output.append(ddp_model(batch.to(rank)))
    
    return outputs

def main():
    world_size = 1
    batch_siz = 64
    mdl = getmodel()
    ds = getdataset()
    dl = DataLoader(ds, batch_size = batch_siz, num_workers=8, drop_last=True)
    
    mp.spawn(run_infer_load,
        args=(world_size, mdl, dl),
        nprocs=world_size,
        join=True)
    
if __name__=="__main__":
    main()
    