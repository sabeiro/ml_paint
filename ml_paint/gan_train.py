import numpy as np
import matplotlib.pyplot as plt
import os, sys
os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION']='false'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
import ml_paint.gan_keras as g_k
import ml_paint.gan_spine as g_s
import importlib
import cv2

opt = g_s.defOpt()
opt = {"isNoise":False,"isHomo":True,"isCat":False,"isLoad":True,"isBlur":False,"nameF":None,"catF":None
       ,"rotate":True,"batch_size":20,"smallD":256,"largeD":320,"zoom":2,"n_img":100
       ,"model_name":"model_pers","baseDir":"/home/sabeiro/tmp/pers/"
       ,"imgDir":"/home/sabeiro/tmp/pers/heim/h/","isS3":False}

fL = g_s.listFile(opt['imgDir'])

if False: # pix2pix
    importlib.reload(g_s)
    importlib.reload(g_k)
    opt["rotate"], opt['isBlur'] = True, True
    gan = g_k.gan_deep(opt)
    for i in range(20):
        fL = g_s.listFile(opt['imgDir'],opt['n_img'])
        X_source, X_target, labelD = g_s.prepImg(fL,opt)
        gan.train(20,X_source,X_target,opt,labelD)
        
    if False:
        disp = g_s.output2pic(X_source)
        disp = g_s.output2pic(X_target)
        plt.imshow(disp);plt.show()
    
if True: # super resoluton
    # opt['smallD'], opt["largeD"] = int(256/4), int(320/4)
    opt['n_img'], opt['batch_size'], opt['zoom'] = 20, 10, 2
    opt['isLoad'] = True
    opt["model_name"] = "model_superRes"
    importlib.reload(g_s)
    importlib.reload(g_k)
    gan = g_k.superRes2(opt)
    gan.gen_model.summary()
    for i in range(20):
        fL = g_s.listFile(opt['imgDir'],opt['n_img'])
        X_source, X_target, labelD = g_s.superRes(fL,opt)
        gan.train(10,X_source,X_target,opt,labelD)
        
        X_pred = gan.gen_model.predict(X_source[:20])
        disp = (X_source[0]*0.5 + 0.5)*255
        disp = np.array(disp,dtype = np.uint8)
        if opt['rotate']:
            disp = cv2.rotate(disp, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        disp = cv2.resize(disp,dsize=(opt['largeD']*opt['zoom']
                                      ,opt['smallD']*opt['zoom']),interpolation=cv2.INTER_CUBIC)
        disp = cv2.cvtColor(disp,cv2.COLOR_RGB2BGR)
        cv2.imwrite(gan.baseDir+"/%05d.jpg"%301,disp)
        gan.save_image(X_target,302,gan.baseDir)
        gan.save_image(X_pred,303,gan.baseDir)

    
        
    disp = g_s.output2pic(X_target)
    importlib.reload(g_s)
    disp = g_s.readPic(fL[0])
    plt.imshow(disp);plt.show()

    
if False: # debug section
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


