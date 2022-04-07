import numpy as np
import cv2, os, re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import matplotlib
import multiprocessing
import psutil
import gc
import face_recognition
gc.isenabled() 
  
baseDir = "/home/sabeiro/lav/tmp/gan/"
baseDir = "/home/sabeiro/lav/src/gen/"
wrtDir = "/home/sabeiro/tmp/pers/"
imgDir = "/home/sabeiro/Pictures/pers/"
face_cascade = cv2.CascadeClassifier(baseDir + 'haar/haarcascade_frontalface_default.xml'); suf = "def"
net = cv2.dnn.readNetFromCaffe("haar/deploy.prototxt","haar/res10_300x300_ssd_iter_140000_fp16.caffemodel")

#face_cascade = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

#face_cascade = cv2.CascadeClassifier(baseDir + 'haar/haarcascade_profileface.xml')
#face_cascade = cv2.CascadeClassifier(baseDir + 'haar/haarcascade_frontalface_alt.xml'); suf = "alt"
eye_cascade = cv2.CascadeClassifier(baseDir + 'haar/haarcascade_eye.xml')
ratio = 4./3.
width = 360
# ratio = 1
# width = 256
height = int(width*ratio)

def findFaceCV(face_cascade,img):
    iw, ih = img.shape[0], img.shape[1]
    faceL = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5
                                          ,minSize=(60,60),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faceL:
        dw = int(w*0.5)
        dh = int(h*0.9)
        dh = int(dw*ratio)
        w = w + dw
        h = int(w*ratio)
        x = max(0,x - int(0.5*dw))
        y = max(0,y - int(0.75*dh))

    return faceL

def findFace(net,img,conf_threshold=0.7):
    frameOpencvDnn = img.copy()
    faceL = []
    w, h = img.shape[0], img.shape[1]
    l = [w,h,w,h]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            box = []
            for j in range(4):
                box.append(detections[0,0,i,3+j]*l[j])
            faceL.append(box)
    return faceL


def cutFace(img):
    resL = []
    faces = []
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.CV_8U)
    gray = np.uint8(gray)
    try: faceL = findFace(face_cascade,img)
    except: return resL
    print("Found %d faces" % (len(faceL)))
    for (x,y,w,h) in faceL:
        dw = int(w*0.5)
        dh = int(dw*ratio)
        w = w + dw
        h = int(w*ratio)
        x = max(0,x - int(0.5*dw))
        y = max(0,y - int(0.75*dh))
        roi_color = img[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print("Found %d faces - %d eyes" % (len(faces),len(eyes)))
        res = cv2.resize(roi_color,dsize=(width,height),interpolation=cv2.INTER_CUBIC)
        resL.append(res)
        #if len(eyes) == 0: continue
        # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        del res
    del faces, gray
    gc.collect()
    return resL

def procImg(f):
    load1, load5, load15 = psutil.getloadavg()  
    cpu_usage = (load15/os.cpu_count()) * 100
    print("The CPU usage is :.1%f RAM: %.f " % (cpu_usage, psutil.virtual_memory()[2]))
    try:
        #img = mpimg.imread(imgDir + d + "/" + f)
        img = cv2.imread(imgDir + d + "/" + f)
        if len(img.shape) == 3:
            if img.shape[2] == 4: img = img[:,:,:3]
            #img = Image.open(imgDir + d + "/" + f)
    except:
        print('could not open %s' % (f))
        return 0
    resL = cutFace(img)
    if len(resL) == 0: return 0
    for i,res in enumerate(resL):
        cv2.imwrite(wrtDir+"/face/"+d+"-"+suf+"_"+str(i)+"_"+f, res)
        # plt.clf()
        # fig = plt.figure(frameon=False)
        # fig.set_size_inches(width,height)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # ax.imshow(res, aspect=1.);
        # fig.savefig(wrtDir+"/face/"+d+"_"+suf+"_"+str(i)+"_"+f, dpi=1)
        # plt.close(fig)
        # plt.close('all')
        # plt.clf()
        # del fig, ax
        # gc.collect()
    del img
    gc.collect()
    return 1

dL = os.listdir(imgDir)
#dL = ['anìma','sab','odelia']
for d in dL:
    print(d)
    fL = os.listdir(imgDir + "/" + d)
    #for f in fL: procImg(f)
    pool = multiprocessing.Pool(10)
    results = pool.map(procImg, fL)
    pool.close()
    pool.join()


if False: #debug single
    dL = os.listdir(imgDir)
    d = dL[5]
    fL = os.listdir(imgDir + "/" + d)
    f = fL[0]
    img = cv2.imread(imgDir+"/"+d+"/"+f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img);plt.show()

    resL = cutFace(img)

    fig, axs = plt.subplots(2, 3)
    count = 0
    for i in range(2):
        for j in range(3):
            cv2.imwrite(gan.baseDir+"/%05d.jpg"%count,resL[count])

            axs[i,j].imshow(resL[count])
            axs[i,j].axis('off')
            count += 1
            plt.tight_layout(pad=0.)
            plt.show()


    
    
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = img[:,:,:3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = (255*gray).astype(int)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    plt.imshow(gray)
    plt.show()
    procImg(f)
    

    
    d = 'anìma'
    fL = os.listdir(imgDir + "/" + d)
    f = fL[2]
    f = [x for x in fL if re.search("20211110",x)][0]
    img = cv2.imread(imgDir + d + "/" + f)
    # plt.imshow(img);    plt.show()
    # plt.imshow(gray);    plt.show()
    # plt.imshow(res);    plt.show()
    # procImg(f)
    resL = cutFace(img)
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = img[:,:,:3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = (255*gray).astype(int)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    plt.imshow(gray)
    plt.show()
    procImg(f)

#p = multiprocessing.Process(target=worker)
#jobs.append(p)
#p.start()
# pool = Pool(6)
# segments = [(stations_with_coords, cilacs_map, id_cilac_map, start_index) for start_index in range(0, len(stations_with_coords), BUFFER)]
# cell_map_distance = pool.map(get_cilacs_to_segment, segments)
            
# if True:
#     plt.imshow(img)
#     plt.show()

    
    # plt.imshow(roi_color)
    # plt.show()
    
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




