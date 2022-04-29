import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, sys
import cv2
os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import ml_paint.gan_keras as g_k
import ml_paint.gan_spine as g_s
import importlib

opt = g_s.defOpt()
opt = {"isNoise":False,"isHomo":True,"isCat":False,"isLoad":True,"isBlur":False,"nameF":None,"catF":None
       ,"rotate":True,"batch_size":20,"smallD":256,"largeD":320,"zoom":4,"n_img":100
       ,"model_name":"model_pers","baseDir":"/home/sabeiro/tmp/pers/"
       ,"imgDir":"/home/sabeiro/tmp/pers/heim/h/"}

fL = g_s.listFile(opt['imgDir'])
f = "sab-0_ProfiloGoeRit.jpg"
#f = "fra-def_0_20150216_123121.jpg"
f = fL[0]

if False:
    gan = g_k.gan_deep(opt)
    n_step = len(fL)//opt['batch_size']
    for i in range(n_step):
        fL1 = fL[i*n_step:(i+1)*n_step]
        X_source, X_target, labelD = g_s.prepImg(fL1,opt)
        res = gan.gen_model.predict(X_source)
        for j in range(opt['batch_size']):
            gan.save_image(res[j:],i*n_step+j,gan.baseDir+'/output/')
    print("pictures saved")

if True:
    opt["imgDir"] = "/home/sabeiro/tmp/pers/sel/sel_mont/"
    opt["model_name"] = "model_superRes"
    opt['n_img'], opt['batch_size'], opt['zoom'] = 20, 20, 2
    fL = g_s.listFile(opt['imgDir'])
    importlib.reload(g_s)
    importlib.reload(g_k)
    gan = g_k.superRes2(opt)
    n_step = len(fL)//opt['batch_size']
    for i in range(n_step):
        fL1 = fL[i*n_step:(i+1)*n_step]
        X_source, X_target, labelD = g_s.superRes(fL1,opt)
        res = gan.gen_model.predict(X_source)
        for j in range(len(res)):
            gan.save_image(res[j:],i*n_step+j,gan.baseDir+'/output/')
    print("pictures saved")
