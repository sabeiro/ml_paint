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
       ,"rotate":True,"batch_size":64
       ,"model_name":"model_pers","baseDir":"/home/sabeiro/tmp/pers/"}

importlib.reload(g_s)
X_source, X_target, labelD = g_s.prepImg(opt['baseDir']+'/mont/h/',opt['isBlur'],nameF=opt['name'])
opt['img_shape'] = X_source[0].shape
importlib.reload(g_s)
importlib.reload(g_k)
gan = g_k.gan_deep(opt)
gan.train(200,X_source,X_target,opt,labelD)

if False:
    f = "sab-0_ProfiloGoeRit.jpg"
    f = "fra-def_0_20150216_123121.jpg"
    inp = g_s.pic2input(opt['baseDir']+'/img/' + f)
    img = g_s.readPic(opt['baseDir']+'/img/' + f)

    opt = {"isNoise":False,"isHomo":True,"isCat":False,"isLoad":True,"isBlur":True,"name":"anìma"
       ,"model_name":"model_pers","baseDir":"/home/sabeiro/tmp/pers/","batch_size":64}
    opt['img_shape'] = img.shape
    X_source, X_target, labelD = g_s.prepImg(opt['baseDir']+'/img/',nameF=f,catF=None)
    gan = g_k.gan_deep(opt)
    res = gan.gen_model.predict(X_source)
    disp = g_s.output2pic(res)
    plt.imshow(disp);plt.show()
    gan.save_image(res, 300, gan.baseDir)
    
    img = mpimg.imread(opt['baseDir']+'/img/' + f)
    plt.imshow(img);plt.show()
    import albio.img_proc as i_p
    importlib.reload(i_p)
    imgV, metaD = i_p.imgDec(img)

    opt['img_shape'] = inp.shape
    gan = g_k.gan_deep(opt)
    res = gan.gen_model.predict(inp)
    res = np.array(res*255,dtype = np.uint8)
    #gan.save_image(res, 0, gan.baseDir)
    imgV = [img] + imgV + [res[0]]
    
    fig, axs = plt.subplots(2, 3)
    count = 0
    for i in range(2):
        for j in range(3):
            axs[i,j].imshow(imgV[count])
            axs[i,j].axis('off')
            count += 1
            plt.tight_layout(pad=0.)
    plt.show()


if False:
    opt = {"isNoise":False,"isHomo":True,"isCat":False,"isLoad":True,"isBlur":True,"name":"anìma"
           ,"model_name":"model_res","baseDir":"/home/sabeiro/tmp/pers/","batch_size":64}
    X_small, X_large, labelD = g_s.superRes(opt['baseDir']+'/img/',opt['isBlur'],nameF=opt['name'])
    gan = g_k.superRes(opt)
    gan.train(200,X_small,X_large,opt,labelD)


    
