from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import imageio
import cv2
import random
os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
import matplotlib.image as mpimg
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.models import Input
import albio.img_proc as i_p

def defOpt():
    return {"isNoise":False,"isHomo":True,"isCat":False,"isLoad":True,"isBlur":False,"name":None
            ,"rotate":False
            ,"model_name":"model_pers","baseDir":"/home/sabeiro/tmp/pers/","batch_size":64}

class gan_spine():
    def __init__(self, opt):
        optimizer = Adam(0.0002, 0.5)
        self.history = []
        self.img_shape = opt['img_shape']
        self.channels = opt['img_shape'][-1]
        self.latent_dim = (int(opt['img_shape'][0]/4),int(opt['img_shape'][1]/4),1)
        self.baseDir = opt['baseDir']
        self.kernel = (4,4)
        self.opt = opt
        if opt['isLoad']:
            self.load_gen_model(optimizer)
            self.load_disc_model(optimizer)
            print('model loaded')
        else:
            self.build_gen_model(optimizer)
            self.build_disc_model(optimizer)
        self.build_gan(optimizer)

        
    def build_gan(self, optimizer):
        for layer in self.disc_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        gen_in = Input(shape=self.img_shape)
        gen_out = self.gen_model(gen_in)
        disc_out = self.disc_model([gen_in,gen_out])
        # gan = Model(gen_in, disc_output)
        self.gan = Model(gen_in, [disc_out, gen_out])
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.gan.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
        plot_model(self.gan,to_file=self.baseDir+'model_gan/gan_model.png',show_shapes=True,show_layer_names=True)

    def gen_real(self, trainA, trainB, batch_size,isHomo=False):
        ix = random.sample(range(len(trainA)), batch_size)
        jx = random.sample(range(len(trainB)), batch_size)
        if isHomo: jx = ix
        X1, X2 = trainA[ix], trainB[jx]
        return [X1, X2]

    def gen_fake(self, samples):
        X = self.gen_model.predict(samples)
        return X
        
    def gen_noise(self, X_train,batch_size):
        ix =  random.sample(range(len(X_train)),batch_size)
        trainA = np.random.normal(0, 1, (batch_size,) + self.img_shape)
        trainB = X_train[ix]
        return [trainA, trainB]

    def update(self,epoch,label,generated,loss_real,loss_fake,gen_loss):
        print ("******************Epoch {}***************************".format(epoch))
        print ("Loss: real %.4f fake %.4f gen %.4f label %s" % (loss_real,loss_fake,gen_loss,label))
        print ("---------------------------------------------------------")
        self.history.append({"D":loss_real,"G":gen_loss})
        namePref = self.baseDir + "model_gan/"+self.opt['model_name']
        if epoch % 1 == 0:
            self.save_image(generated, epoch, self.baseDir + "/generated/")
            self.plot_loss()
        if (epoch+1) % 10 == 0:
            model_json = self.disc_model.to_json()
            with open(namePref+"_disc.json", "w") as json_file:
                json_file.write(model_json)
            self.disc_model.save_weights(namePref+"_disc.h5")
            model_json = self.gen_model.to_json()
            with open(namePref+"_gen.json", "w") as json_file:
                json_file.write(model_json)
            self.gen_model.save_weights(namePref+"_gen.h5")
            print("Saved model to disk")

    def train(self,n_epoch,X_source,X_target,opt,labelD={}):
        batch_size = opt['batch_size']
        dim_disc = tuple(self.disc_model.layers[-1].output.shape)
        dim_disc = (batch_size,) + dim_disc[1:]
        y_real = np.ones(dim_disc)
        y_fake = np.zeros(dim_disc)
        label = 'all'
        for epoch in range(n_epoch):
            if labelD and opt['isCat']:
                label = random.choice(list(labelD.keys()))
                v = random.choice(labelD[label])
                l = [x == v for x in labelD[label]]
                X_source1 = X_source[l]
            else: X_source1 = X_source
            if opt['isNoise']: [X_realA, X_realB] = self.gen_noise(X_source1, batch_size)
            else: [X_realA, X_realB] = self.gen_real(X_source,X_target,batch_size,opt['isHomo'])
            X_fakeB = self.gen_fake(X_realA)
            loss_real = self.disc_model.train_on_batch([X_realA, X_realB], y_real)
            loss_fake = self.disc_model.train_on_batch([X_realA, X_fakeB], y_fake)
            gen_loss, _, _ = self.gan.train_on_batch(X_realA, [y_real, X_realB])
            self.update(epoch,label,X_fakeB,loss_real,loss_fake,gen_loss)
            
    def save_image(self, generated, epoch, directory):
        disp = 0.5 * generated[0] + 0.5
        if self.opt['rotate']:
            disp = cv2.rotate(disp, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        #disp = cv2.cvtColor(disp*256,cv2.COLOR_RGB2BGR)
        cv2.imwrite(directory+"/%05d.jpg"%epoch,disp*256)
        # height, width, channel = self.img_shape
        # plt.clf()
        # fig = plt.figure(frameon=False)
        # fig.set_size_inches(width,height)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # ax.imshow(generated[0], aspect=1.)
        # fig.savefig(directory + "/%05d.png" % epoch,dpi=1)
        # plt.close()

    def save_image_grid(self, generated, epoch, directory):
        noise = np.random.normal(0, 1, (25,) + self.latent_dim)
        generated = self.gen_model.predict(noise)
        generated = 0.5 * generated + 0.5
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(generated[count, :,:,:])
                axs[i,j].axis('off')
                count += 1
                plt.tight_layout(pad=0.)
        fig.savefig("{}/{}.png".format(directory, epoch))
        plt.close('all')
        plt.clf()
                
    def load_gen_model(self, optimizer):
        namePref = self.baseDir + "model_gan/"+self.opt['model_name']
        json_file = open(namePref + '_gen.json', 'r')
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        json_file.close()
        loaded_model.load_weights(namePref + '_gen.h5')
        self.gen_model = loaded_model
        # print(self.gen_model.summary())
        
    def load_disc_model(self, optimizer):
        namePref = self.baseDir + "model_gan/"+self.opt['model_name']
        json_file = open(namePref + '_disc.json', 'r')
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        json_file.close()
        loaded_model.load_weights(namePref + '_disc.h5')
        self.disc_model = loaded_model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc_model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        self.disc_model.trainable = False
        # print(self.disc_model.summary())

    def receptive_field(self,output_size, kernel_size, stride_size):
        return (output_size - 1) * stride_size + kernel_size

    def predict_noise(self, size):
        noise = np.random.normal(0, 1, (size,) + self.latent_dim)
        return self.gen_model.predict(noise)
        
    def plot_loss(self):
        hist = pd.DataFrame(self.history)
        plt.figure(figsize=(20,5))
        for colnm in hist.columns:
            plt.plot(hist[colnm],label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.savefig(self.baseDir + 'history.png')
        plt.clf()
        plt.close('all')

def readPic(f):
    img = mpimg.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[1]>img.shape[0]: img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img,dsize=(256,320),interpolation=cv2.INTER_CUBIC)
    return img
        
def pic2input(f):
    img = readPic(f)
    img = np.array(img) / 127.5 - 1.
    return np.reshape(img,(1,) + img.shape)

def output2pic(res):
    disp = (res[0]*0.5 + 0.5)*255
    return np.array(disp,dtype = np.uint8)
        
def prepImg(projDir,isBlur=False,nameF=None,catF=None):
    blur_x, blur_y = 0, 0
    fL = os.listdir(projDir)
    X_target, X_filter, labelL = [], [], []
    img = mpimg.imread(projDir + fL[0])
    h, w, l = img.shape
    image_stack = np.ones((2, 2, 18))
    padW = int(abs(480-w)/2)
    padH = int(abs(480-h)/2)
    blur_x, blur_y = min(60,int(w*.5)), min(60,int(h*.5))
    for f in fL:
        rotation = False
        name = f.split(os.path.sep)[-1].split("-")[0]
        if nameF:
            if f != nameF: continue
        if catF:
            if name != catF: continue
        img = readPic(projDir + f)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = img[:,:,:3]
        if len(img.shape) == 2:
            img1 = np.zeros(img.shape + (3,))
            for d in range(3): img1[:,:,d] = img
            img1 = np.array(img1,dtype = np.uint8)
            img = img1
        if img.shape[1]>img.shape[0]: rotation = True
        if isBlur:
            """blur outside face"""
            sub = cv2.GaussianBlur(img,(23, 23), 30)
            h, w, l = img.shape
            x, y = int(w*0.5), int(h*0.4)
            dw, dh = int(blur_x*np.random.uniform()), int(blur_y*np.random.uniform())
            sub_face = img[y-dh:y+dh, x-dw:x+dw]
            sub[y-dh:y+dh, x-dw:x+dw] = sub_face
            img = sub
        #img = np.pad(img,((padH,padH),(padW,padW),(0, 0)),mode='constant',constant_values=0)
        #img = cv2.resize(img,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
        imgV, metaD = i_p.imgDec(img)
        metaD['name'] = name
        metaD['rotation'] = rotation
        X_target.append(img)
        X_filter.append(imgV[1])
        labelL.append(metaD)

    if len(labelL) == 0:
        print("no image found")
        return [], [], {}
    X_target = np.array(X_target) / 127.5 - 1.
    X_filter = np.array(X_filter) / 127.5 - 1.
    labelD = {}
    for k in labelL[0].keys(): labelD[k] = [v[k] for v in labelL]
    return X_filter, X_target, labelD


def superRes(projDir,isHor=False):
    colorL = ['red','blue','green']
    dL = os.listdir(projDir)
    smallD, largeD, zoom = 256, 320, 4
    smallL, largeL = [], []
    for d in dL:
        fL = os.listdir(imgDir + "/" + d)
        for f in fL:
            img = mpimg.imread(projDir + d + "/" + f)
            h, w, l = img.shape
            name = f.split(os.path.sep)[-1].split("-")[0]
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = img[:,:,:3]
            if len(img.shape) != 3:
                print('layer missing')
                continue
            small = cv2.resize(img,dsize=(smallD,longD),interpolation=cv2.INTER_CUBIC)
            large = cv2.resize(img,dsize=(smallD*zoom,longD*zoom),interpolation=cv2.INTER_CUBIC)
            smallL.append(small)
            largeL.append(large)
            r, g, b = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
            bright = np.mean([r,g,b])
            color = colorL[np.argmax([r,g,b])]
            d = {"name":name,"red":r>128,"blue":b>128,"green":g>128,"bright":bright>128,"color":color}
            labelL.append(d)
    X_small = np.array(smallL)/127.5 - 1.
    X_large = np.array(largeL)/127.5 - 1.
    labelD = {}
    for k in labelL[0].keys(): labelD[k] = [v[k] for v in labelL]
    return X_small, X_large, labelD

