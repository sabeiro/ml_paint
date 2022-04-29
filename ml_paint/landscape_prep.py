import numpy as np
import cv2, os, re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import multiprocessing
import psutil, gc
gc.isenabled() 
  
baseDir = "/home/sabeiro/lav/src/gen/"
wrtDir = "/home/sabeiro/tmp/pers/"
imgDir = "/home/sabeiro/Pictures/mont/"
shortL = 256
longL = 320

def procImg(f):
    load1, load5, load15 = psutil.getloadavg()  
    cpu_usage = (load15/os.cpu_count()) * 100
    print("The CPU usage is :.1%f RAM: %.f " % (cpu_usage, psutil.virtual_memory()[2]))
    try:
        #img = mpimg.imread(imgDir + d + "/" + f)
        img = cv2.imread(imgDir + d + "/" + f)
        if len(img.shape) == 3:
            if img.shape[2] == 4: img = img[:,:,:3]
    except:
        print('could not open %s' % (f))
        return 0
    if img.shape[0] > img.shape[1]: 
        res = cv2.resize(img,dsize=(shortL,longL),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(wrtDir+"/heim/v/"+d+"-"+f, res)
    else:
        res = cv2.resize(img,dsize=(longL,shortL),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(wrtDir+"/heim/h/"+d+"-"+f, res)
    del img
    gc.collect()
    return 1

dL = os.listdir(imgDir)
#dL = ['anìma','sab','odelia']
for d in dL:
    print(d)
    fL = os.listdir(imgDir + "/" + d)
    pool = multiprocessing.Pool(10)
    results = pool.map(procImg, fL)
    pool.close()
    pool.join()

