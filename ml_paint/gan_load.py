import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, sys
import cv2
os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import gan_keras as g_k
import gan_spine as g_s
import importlib

opt = g_s.defOpt()
opt = {"isNoise":False,"isHomo":True,"isCat":False,"isLoad":True,"isBlur":False,"name":None
       ,"rotate":False
       ,"model_name":"model_pers","baseDir":"/home/sabeiro/tmp/pers/","batch_size":64}

f = "sab-0_ProfiloGoeRit.jpg"
#f = "fra-def_0_20150216_123121.jpg"
f = None
X_source, X_target, labelD = g_s.prepImg(opt['baseDir']+'/img/',nameF=f,catF=None)
opt['img_shape'] = X_source[0].shape
gan = g_k.gan_deep(opt)
res = gan.gen_model.predict(X_source)
for i in range(len(res)):
    gan.save_image(res[i:], 300+i, gan.baseDir+'/output/')
print("pictures saved")
