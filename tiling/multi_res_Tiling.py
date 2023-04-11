import requests
from .constants import getbraindetails, get_iip_rgn
import os
from collections import namedtuple
import glymur
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import io
from skimage.io import imread
# from .utils import create_mmap, jp2_to_tif, 
from .utils import get_multiproc_plan
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from datetime import datetime
from joblib import Parallel, delayed
import pickle

Point = namedtuple('Point','x,y')
Span = namedtuple('Span','w,h')
Box = namedtuple('Box','point,span')
Extent = namedtuple('Extent','point1,point2')

def to_shape(span):
    return np.array([span.h,span.w])

def to_extent(box): 
    pt2 = Point(box.point.x+box.point.w, box.point.y+box.point.h)
    return Extent(box.point,pt2)

def to_box(ext):
    span = Span(ext.point2.x-ext.point1.x, ext.point2.y-ext.point1.y)
    return Box(ext.point1,span)

def to_slice(ext,step=1):
    """can directly index as arr[to_slice(ext)] - equiv to 
        arr[ext.point1.y:ext.point2.y,ext.point1.x:ext.point2.x,:]"""
    rsl = slice(int(ext.point1.y),int(ext.point2.y),step)
    csl = slice(int(ext.point1.x),int(ext.point2.x),step)
    return rsl,csl 


# !pwd

class SectionProxy:
    
    def __init__(self, brainid=15, seriesType="NISSL", secnumber=1708, iscompressed=False):
        
        self.brainid = brainid
        self.seriesType = seriesType
        self.secnumber = secnumber
        
        # uses getBrainViewerDetails api call to get jp2path, width and height of requested section
        # alternative: can use iip iiif info.json to find width and height of jp2
    
        if seriesType in ('NISSL','HEOS'):
            
            self.secinfo = getbraindetails(brainid,seriesType,secnumber)

            if not iscompressed:
                # provide lossless path 
                self.secinfo['jp2Path'] = self.secinfo['jp2Path'].replace('compressed','lossless')
            
        self.modes = {'jp2':False,'iip':True,'mmap':False,'tif':False}
        
        
    def check_local_jp2(self,imgroot = './storage/jp2cache'):
        path,jp2base = os.path.split(self.secinfo['jp2Path'])

        if os.path.exists(imgroot+'/'+jp2base):
            self.modes['jp2'] = True
            self.jp2path = imgroot+'/'+jp2base
        else:            
            self.jp2path = None
            self.modes['jp2'] = False
        print(imgroot + '/' + jp2base)
        return self.modes['jp2']

    def make_local_copy(self,destdir = './storage/jp2cache'):
        bn = os.path.basename(self.jp2path)
        assert not os.path.exists(destdir+'/'+bn)
        shutil.copy(self.jp2path,destdir)
        return self.check_local_jp2(destdir)

    def check_mmap(self, mmapdir = './storage/mmapdir'):
        loc,bn = os.path.split(self.jp2path)
        namepart = ".".join(bn.split('.')[:-1])
        
        mmapfilename = mmapdir+'/'+namepart+'.dat'
        infoname = mmapfilename.replace('.dat','_info.pkl')
        matched = False
        shp = to_shape(self.span())
        if os.path.exists(infoname):
            info = pickle.load(open(infoname,'rb'))
            if info['shape'][0]==shp[0] and info['shape'][1]==shp[1] and info['fname']==self.jp2path and os.path.exists(mmapfilename):
                matched = True
            else: print(infoname,info)
        if matched:
            self.modes['mmap'] = True
            self.mmappath = mmapfilename
        
        return self.modes['mmap']
    
    def check_tif(self, tifdir = ''):
        loc,bn = os.path.split(self.jp2path)
        namepart = ".".join(bn.split('.')[:-1])
        tifname = tifdir+'/'+namepart+'.tif'
        if os.path.exists(tifname):
            self.tifpath = tifname
            self.modes['tif']=True
            
        return self.modes['tif']
        
    def kdu_expand(self,outdir= ''):
        if not self.check_tif(outdir):
            assert self.modes['jp2'] is True
            libdir = kdubase+'/lib/Linux-x86-64-gcc'
            bindir = kdubase+'/bin/Linux-x86-64-gcc'

            bn = os.path.basename(self.jp2path)
            namepart = ".".join(bn.split('.')[:-1])
            outname = outdir+'/'+namepart+'.tif'
            start = datetime.now()
            ret = os.system(f'LD_LIBRARY_PATH={libdir} {bindir}/kdu_expand -i %s -o %s' % (self.jp2path,outname))
            assert ret==0
            self.tifpath = outname
            self.modes['tif']=True
            print(datetime.now()-start)
        return self.check_tif(outdir)
            
    def span(self):
        return Span(self.secinfo['width'],self.secinfo['height'])
         

    def __str__(self):
        return (
            str(self.secinfo)+
            str(
                {
                    'brainid':self.brainid, 
                    'seriesType':self.seriesType,
                    'secnumber':self.secnumber,
                    'modes':self.modes
                }
            )
        )

def get_tilespan(resolution):
    # assume 0.5 um per pix, and 2mm span at native
    # refer gsheet
    # tile size in um = tilespan.w*resolution ; e.g 2500x0.5, 2000x4, 500x32
    # tiling of 16,32,64 should look the same (just halving npix)
    
    if resolution in (0,1,2): # 40x, 20x, 10x
        return Span(2500,2500)
    if resolution == 4: # 5x
        return Span(2000,2000)
    npix = {8:1500,16:1000,32:500,64:250} # 2.5, 1.25, 0.625, 0.3125
    
    return Span(npix[resolution],npix[resolution])


def box_to_rgn(box, span):    
    xrel = box.point.x/span.w
    yrel = box.point.y/span.h
    wrel = box.span.w/span.w
    hrel = box.span.h/span.h
    return xrel,yrel,wrel,hrel

class TileAccessor:
    def __init__(self,sectionproxy, resolution = 0, tilespan = 'auto', padding=0, use_iip=True ):
        self.sectionproxy = sectionproxy 
        self._jp2handle = None
        self._tifdata = None
        self._mmaphandle = None
        
        assert padding >= 0
        self.padding = padding
        if not use_iip:
            if sectionproxy.modes['tif']:
                self._tifdata = imread(sectionproxy.tifpath)
                shp = self._tifdata.shape

            elif sectionproxy.modes['mmap']:

                infoname = self.sectionproxy.mmappath.replace('.dat','_info.pkl')
                info = pickle.load(open(infoname,'rb'))
                shp = info['shape']
                self._mmaphandle = np.memmap(self.sectionproxy.mmappath,dtype='uint8',mode='r',shape=shp )


            elif sectionproxy.modes['jp2']:
                #glymur.set_option('lib.num_threads',48) # originally 12 on ml01, 48 on analyse2
                glymur.set_option('print.codestream',False)
                glymur.set_option('print.xml',False)

                self._jp2handle = glymur.Jp2k(sectionproxy.jp2path)
                shp = self._jp2handle.shape # nr,nc,3
        
        else:
            span = self.sectionproxy.span()
            shp = span.h,span.w # nr,nc

        self.resolution = resolution # um ; 0=> native
        self.imagespan = Span(shp[1],shp[0])
        self.origimgspan = Span(shp[1],shp[0])
        self.dec_factor = 1
        if resolution > 0:
            self.dec_factor = resolution/0.5 # FIXME: assumes native=0.5
            self.imagespan = Span(shp[1]//self.dec_factor,shp[0]//self.dec_factor)
        
        if tilespan == 'auto':
            self.tilespan = get_tilespan(resolution)
        else:
            assert type(tilespan)==Span
            self.tilespan = tilespan
            
        self.ntiles_c = round(self.imagespan.w/self.tilespan.w) # FIXME: was ceil, changing to round for compat with ui
        self.ntiles_r = round(self.imagespan.h/self.tilespan.h) # not actually used anywhere - only print
        self.ntiles = self.ntiles_r * self.ntiles_c
        
        self._tnail = None

    def _wsi(self):
        if self._tifdata is None and self.sectionproxy.modes['jp2']:
            self._tifdata=self._jp2handle[:]
        
    def get_thumbnail(self,res=128):
        assert 2**int(math.log(res,2))==res, "res expected as 2^n"
        if self._tnail is not None:
            return self._tnail, res
        
        if self._tifdata is not None:
            self._tnail = self._tifdata[::res,::res,:]
        
        elif self._mmaphandle is not None:
            self._tnail = self._mmaphandle[::res,::res,:]
            
        elif self._jp2handle is not None:
            # 64 um pixels
            self._tnail = self._jp2handle[::res,::res,:]
             
        else:
            box = Box(Point(0,0),self.origimgspan)
            rgn = box_to_rgn(box,self.origimgspan)

            self._tnail,_ = get_iip_rgn(self.sectionproxy.secinfo['jp2Path'],box.span.w//res,box.span.h//res,rgn)
            
        return self._tnail, res
        

    def get_tilenum(self,x,y):
        assert y>=0 and y<self.imagespan.h
        assert x>=0 and x<self.imagespan.w
        assert self.tilespan is not None

        tile_r,tile_c = y//self.tilespan.h, x//self.tilespan.w
        return tile_r*self.ntiles_c+tile_c
        
    def get_tile_extent(self,tilenum):
        if tilenum>=self.ntiles:
            # assert tilenum<self.ntiles, f"failed tilenum < self.ntiles ({tilenum},{self.ntiles})"
            raise StopIteration
        assert tilenum>=0
        
        tile_r = tilenum//self.ntiles_c
        tile_c = tilenum % self.ntiles_c
        tl = Point(self.tilespan.w*tile_c,self.tilespan.h*tile_r)
        br = Point(min(tl.x+self.tilespan.w,self.imagespan.w),min(tl.y+self.tilespan.h,self.imagespan.h))
        
        return Extent(tl,br)
    
    def get_padded_extent(self,ext):
        padding_r = self.padding
        padding_c = self.padding
        
        nc = self.imagespan.w
        nr = self.imagespan.h

        r1 = ext.point1.y
        r2 = ext.point2.y
        c1 = ext.point1.x
        c2 = ext.point2.x

        if r2-r1 < self.tilespan.h+2*self.padding:
            padding_r = (self.tilespan.h+2*self.padding-(r2-r1))//2
            
        if c2-c1 < self.tilespan.w+2*self.padding:
            padding_c = (self.tilespan.w+2*self.padding-(c2-c1))//2
            
        mirror_top = 0
        if r1 - padding_r < 0:
            mirror_top = -(r1 - padding_r)
        else:
            r1 = r1-padding_r

        mirror_left = 0
        if c1 - padding_c < 0:
            mirror_left = -(c1 - padding_c)
        else:
            c1 = c1-padding_c

        mirror_bot = 0
        if r2 + padding_r >= nr:
            mirror_bot = (r2 + padding_r) - nr
        else:
            r2 = r2+padding_r

        mirror_right = 0
        if c2 + padding_c >= nc:
            mirror_right = (c2 + padding_c) - nc
        else:
            c2 = c2+padding_c
            
        df = int(self.dec_factor)    
        ext2 = Extent(Point(c1*df,r1*df),Point(c2*df,r2*df))
        region = Extent(Point(c1-mirror_left,r1-mirror_top), Point(c2+mirror_right,r2+mirror_bot))
        return ext2, region, (mirror_top, mirror_left, mirror_bot, mirror_right)

    def __len__(self):
        return self.ntiles
            
    def __getitem__(self,tilenum):
        
        # print('access %d' % tilenum)
        ext = self.get_tile_extent(tilenum)
        
        ext2, region, mirrorvals = self.get_padded_extent(ext)
        # print(to_box(ext),to_box(ext2),to_box(region),mirrorvals)
        
        imgurl = None

        df = int(self.dec_factor)
        
        if self._tifdata is not None:
            arr = self._tifdata[to_slice(ext2,df)]
        elif self._mmaphandle is not None:
            arr = self._mmaphandle[to_slice(ext2,df)]
        elif self._jp2handle is not None:
            # arr = self.jp2[int(r1)*df:int(r2)*df:df,int(c1)*df:int(c2)*df:df]
            arr = self._jp2handle[to_slice(ext2,df)]

        else:
            
            box = to_box(ext2)
            rgn = box_to_rgn(box,self.origimgspan)
            
            arr,imgurl = get_iip_rgn(self.sectionproxy.secinfo['jp2Path'],box.span.w//df,box.span.h//df,rgn)
            
        (mirror_top, mirror_left, mirror_bot, mirror_right) = mirrorvals
        if mirror_top > 0:
            arr = np.pad(arr,[(mirror_top,0),(0,0),(0,0)],mode='reflect')
        if mirror_left > 0:
            arr = np.pad(arr,[(0,0),(mirror_left,0),(0,0)],mode='reflect')
        if mirror_bot> 0:
            arr = np.pad(arr,[(0,mirror_bot),(0,0),(0,0)],mode='reflect')
        if mirror_right > 0:
            arr = np.pad(arr,[(0,0),(0,mirror_right),(0,0)],mode='reflect')
        
        return arr, region, imgurl

    def __str__(self):
        return str(
                {
                    'proxy':str(self.sectionproxy), 
                    'resolution': self.resolution,
                    'imagespan':self.imagespan,
                    'tilespan':self.tilespan,
                    'ntiles': self.ntiles, 
                    'ntiles_c': self.ntiles_c,
                    'ntiles_r': self.ntiles_r
                }
            )

    def display_tiling(self,res=128):

        plt.figure(figsize=(8,8))
        tnail,_ = self.get_thumbnail(res)
        plt.imshow(tnail)
        # plt.imshow(data)
        plt.gca().set_xticks(np.arange(0,tnail.shape[1],self.tilespan.w//(res/self.dec_factor)))
        plt.gca().set_yticks(np.arange(0,tnail.shape[0],self.tilespan.h//(res/self.dec_factor)))

        plt.grid(True)

def workerfunc(accessor,tilenum):
    return accessor[tilenum]

class SectionMemmap:
    def __init__(self,sectionproxy,mmapdir='./storage/mmapdir',force=False):
        assert sectionproxy.modes['jp2'] is True
        self.sectionproxy = sectionproxy
        loc,bn = os.path.split(self.sectionproxy.jp2path)
        namepart = ".".join(bn.split('.')[:-1])
        if hasattr(self.sectionproxy,'mmappath'):
            self.mmapfilename = self.sectionproxy.mmappath
        else:
            self.mmapfilename = mmapdir+'/'+namepart+'.dat'

        self.infoname = self.mmapfilename.replace('.dat','_info.pkl')
        
        matched = self.sectionproxy.check_mmap(mmapdir)
        
        self.create_pending = False
        if not os.path.exists(self.infoname) or not matched or force:
            self.create_pending = True
            print('call "create"')
            
        
    def create(self,tilespan=Span(4096,4096)):
        assert self.create_pending
        shp = to_shape(self.sectionproxy.span())
        #print(shp)
        self.shape = (shp[0],shp[1],3)

        info = {'dtype':'uint8', 'shape':self.shape,'mmname':self.mmapfilename,'fname':self.sectionproxy.jp2path}
        
        pickle.dump(info,open(self.infoname,'wb'))
        self.handle = np.memmap(self.mmapfilename,dtype='uint8',mode='w+',shape=self.shape )
        
        accessor = TileAccessor(self.sectionproxy,tilespan=tilespan,padding=0,use_iip=False)
        
        plan = get_multiproc_plan(accessor.ntiles,minwork=10)
        print(plan)
        
        workerfunc2 = partial(workerfunc, accessor)
        start=datetime.now()
        print('load started...',end="")
        with ProcessPoolExecutor(max_workers=plan.nworkers) as executor:
            for data,extent,url in executor.map(workerfunc2,range(plan.worksize),chunksize=plan.rounds):
                #print(extent)
                rsl,csl = to_slice(extent) #get slice according to the extent
                #print(rsl, csl)
                self.handle[rsl,csl,:]=data 
        print('loaded. syncing...',end="")
        loadend = datetime.now()
        self.handle.flush()
        flushend = datetime.now()
        print('done')
        self.sectionproxy.mmappath = self.mmapfilename
        self.sectionproxy.modes['mmap'] = True
        self.create_pending = False
        return loadend-start,flushend-loadend # loadtime, flushtime

class TileIterator:
    # work in progress
    # to simulate chunking in processpool
    def __init__(self,accessor,offset = 0, limit = None):
        
        self.accessor = accessor
        self.start = 0
        self.current = None
        self.end = len(accessor)
    
        assert offset >=0 
        self.start = offset -1
        if limit is not None:
            assert limit > 0
            self.end = min(self.end, offset + limit)
            
    def __iter__(self):
        self.current = -1
        return self
    
    def  __next__(self):
        if self.current is None: 
            self.current=0
        else:
            self.current+=1
            
        if  self.current < self.end:      
            return self.accessor[self.current]
        else:
            raise StopIteration
            
    def __len__(self):
        return self.end-self.start
    
    def available(self):
        if self.current is None: return self.__len__()
        return self.end - self.current-1

