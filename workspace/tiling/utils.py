import os
import numpy as np
from skimage.io import imread
import pickle
from multiprocessing import cpu_count
from collections import namedtuple

def jp2_to_tif(jp2filename,outname):
    # outname = 'temp/test.tif'
    assert not os.path.exists(outname)
    assert os.path.exists(jp2filename)
    # FIXME: kdu.so location instead of $PWD
    res = os.system('LD_LIBRARY_PATH=$PWD ./kdu_expand -i %s -o %s' % (jp2filename,outname))
    assert res == 0
    
def create_mmap(arr,fname,outdir):
    path,basename = os.path.split(fname)
    prefix,ext = os.path.splitext(basename)
    pklfile = "%s/%s_info.pkl"%(path,prefix)
    info = {'dtype':arr.dtype, 'shape':arr.shape,'fname':fname}
    pickle.dump(info,open(pklfile,'wb'))
    mmname = "%s/%s_arr.dat"%(path,prefix)
    fp = np.memmap(mmname, dtype=arr.dtype, mode='w+', shape=arr.shape)
    fp[:]=arr[:]
    fp.flush()


MultiprocPlan = namedtuple('MultiprocPlan','worksize,nworkers,perworker,rounds,minwork')

def get_multiproc_plan(worksize,minwork=50):
    
    assert minwork>0 
    minwork = min(minwork, worksize)
    
    maxcnt = cpu_count()
    
    for factor in (1/2, 2/3, 3/4, 4/5, 5/6, 6/7, 7/8):

        nworkers = int(maxcnt*factor)
    
        perworker = worksize//nworkers

        rounds = perworker//minwork
        
        while rounds < 1 and nworkers > 1:
            nworkers=nworkers//2
            perworker = worksize//nworkers

            rounds = perworker//minwork
        
        if rounds < 1:
            rounds = 1
            break
            
        if rounds > 5: # need more workers
            continue
        else:
            break
    return MultiprocPlan(worksize,nworkers,perworker,rounds,minwork)


def dict_filter(dictobj,cnt=None,**kwargs):
    outelts = []
    for elt in dictobj:
        matched=True
        for key,value in kwargs.items():
            if elt[key]!=value:
                matched=False
                break
        if matched:
            outelts.append(elt)
        if cnt is None:
            continue
        if len(outelts)==cnt:
            break
                
    return outelts
