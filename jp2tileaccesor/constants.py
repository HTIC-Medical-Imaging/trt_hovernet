iipserver = 'http://apollo2.humanbrain.in:9081/fcgi-bin/iipsrv.fcgi'
restAPI = "http://apollo2.humanbrain.in:8000"

iipurl = {
    "info.json":'?IIIF=%s/info.json',
    "fif.jtl": '?FIF=%s&WID=1024&GAM=1&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL=%s,%s',
    "fif.jtl.0":'?FIF=%s&WID=1024&GAM=1&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL=0,0',
    "fif.rgn":'?FIF=%s&WID=%d&HEI=%d&RGN=%f,%f,%f,%f&CVT=jpeg',
}
apiurl = {
    "brainDetails":"/GW/getBrainViewerDetails/IIT/V1/SS-%s:-1:-1"
}

import requests
from .utils import dict_filter
import io
from PIL import Image
import numpy as np
import json
import os

def getbraindetails(brainid,seriesType,secnumber=None):
    url=restAPI+apiurl['brainDetails']%(brainid)
    if not os.path.exists('15_10.json'):
        res = requests.get(url).json()
    else:
        res = json.load(open('15_10.json'))
    out = res['thumbNail'][seriesType]

    if secnumber is not None:
        out = dict_filter(out,1,position_index=secnumber)[0]

    return out

def get_iip_rgn(jp2path,width,height,rgn):
    imgurl = iipserver+iipurl['fif.rgn'] % (jp2path,
                width,height,rgn[0],rgn[1],rgn[2],rgn[3])
    # print(imgurl)
    imgbytes = io.BytesIO(requests.get(imgurl).content)
    return np.array(Image.open(imgbytes)), imgurl


