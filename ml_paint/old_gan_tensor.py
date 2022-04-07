from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import imageio
import cv2
import multiprocessing

os.environ['LAV_DIR'] = '/home/sabeiro/lav'
import matplotlib.image as mpimg
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = "/home/sabeiro/lav/tmp/gan/"

class ImageHelper(object):
    def save_image(self, generated, epoch, directory):
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(generated[count, :,:,:])
                axs[i,j].axis('off')
                count += 1
        plt.tight_layout(pad=0.)
        fig.savefig("{}/{}.png".format(directory, epoch))
        plt.close()
        

class DCGAN():
    def __init__(self, image_shape, gen_input_dim, image_hepler, img_channels):
        optimizer = Adam(0.0002, 0.5)
        self._image_helper = image_hepler
        self.img_shape = image_shape
        self.gen_input_dim = gen_input_dim
        self.channels = img_channels
        self._build_gen_model()
        self._build_disc_model(optimizer)
        self._build_gan(optimizer)

    def train(self, epochs, train_data, batch_size):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []
        for epoch in range(epochs):
            #  Train Discriminator
            batch_indexes = np.random.randint(0, train_data.shape[0], batch_size)
            batch = train_data[batch_indexes]
            noise = np.random.normal(0, 1, (batch_size, self.gen_input_dim))
            generated = self.gen_model.predict(noise)
            loss_real = self.disc_model.train_on_batch(batch, real)
            loss_fake = self.disc_model.train_on_batch(generated, fake)
            disc_loss = 0.5 * np.add(loss_real, loss_fake)
            #  Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.gen_input_dim))
            gen_loss = self.gan.train_on_batch(noise, real)
            # Plot the progress
            print ("---------------------------------------------------------")
            print ("******************Epoch {}***************************".format(epoch))
            print ("Discriminator loss: {}".format(disc_loss[0]))
            print ("Generator loss: {}".format(gen_loss))
            print ("---------------------------------------------------------")
            history.append({"D":disc_loss[0],"G":gen_loss})
            # Save images from every hundereth epoch generated images
            if epoch % 100 == 0:
                self._save_images(epoch)
                model_json = self.disc_model.to_json()
                with open(baseDir + "model_disc.json", "w") as json_file:
                    json_file.write(model_json)
                self.disc_model.save_weights(baseDir + "model_disc.h5")
                model_json = self.gen_model.to_json()
                with open(baseDir + "model_gen.json", "w") as json_file:
                    json_file.write(model_json)
                self.gen_model.save_weights(baseDir + "model_gen.h5")
                print("Saved model to disk")
                self._plot_loss(history)
    
    def _build_gen_model(self):
        gen_input = Input(shape=(self.gen_input_dim,))
        gen_sequence = Sequential(
                [Dense(128 * 7 * 7, activation="relu", input_dim=self.gen_input_dim),
                 Reshape((7, 7, 128)),
                 UpSampling2D(),
                 Conv2D(128, kernel_size=3, padding="same"),
                 BatchNormalization(momentum=0.8),
                 Activation("relu"),
                 UpSampling2D(),
                 Conv2D(64, kernel_size=3, padding="same"),
                 BatchNormalization(momentum=0.8),
                 Activation("relu"),
                 Conv2D(self.channels, kernel_size=3, padding="same"),
                 Activation("tanh")])
        
        gen_sequence = Sequential(
                [Dense(128, input_dim=self.gen_input_dim),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(256),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(512),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(np.prod(self.img_shape), activation='tanh'),
                 Reshape(self.img_shape)])
    
        gen_output_tensor = gen_sequence(gen_input)       
        self.gen_model = Model(gen_input, gen_output_tensor)
        print(self.gen_model.summary())
        
    def _build_disc_model(self, optimizer):
        disc_input = Input(shape=self.img_shape)
        disc_sequence = Sequential(
                [Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(64, kernel_size=3, strides=2, padding="same"),
                 ZeroPadding2D(padding=((0,1),(0,1))),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(128, kernel_size=3, strides=2, padding="same"),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(256, kernel_size=3, strides=2, padding="same"),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Flatten(),
                 Dense(1, activation='sigmoid')])

        disc_tensor = disc_sequence(disc_input)
        self.disc_model = Model(disc_input, disc_tensor)
        self.disc_model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.disc_model.trainable = False
        print(self.disc_model.summary())
    
    def _build_gan(self, optimizer):
        real_input = Input(shape=(self.gen_input_dim,))
        gen_output = self.gen_model(real_input)
        disc_output = self.disc_model(gen_output)        
        
        self.gan = Model(real_input, disc_output)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def _save_images(self, epoch):
        generated = self._predict_noise(25)
        generated = 0.5 * generated + 0.5
        self._image_helper.save_image(generated, epoch, baseDir + "/generated-dcgan/")
    
    def _predict_noise(self, size):
        noise = np.random.normal(0, 1, (size, self.gen_input_dim))
        return self.gen_model.predict(noise)
        
    def _plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(20,5))
        for colnm in hist.columns:
            plt.plot(hist[colnm],label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.savefig(baseDir + 'history.png')
        plt.clf()


fL = os.listdir(baseDir + "/face/")
XL = []
for f in fL:
    img = mpimg.imread(baseDir + "/face/" + f)
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = img[:,:,:3]
    if len(img.shape) != 3:
        print('layer missing')
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if True:
        """blur outside face"""
        sub = cv2.GaussianBlur(img,(23, 23), 30)
        h, w, l = img.shape
        x, y = int(w*0.5), int(h*0.4)
        dw, dh = int(60*np.random.uniform()), int(60*np.random.uniform())
        sub_face = img[y-dh:y+dh, x-dw:x+dw]
        sub[y-dh:y+dh, x-dw:x+dw] = sub_face
        img = sub
    if False:
        plt.imshow(img)
        plt.show()
    #gray = gray[:28,:28]
    XL.append(img)

X = np.array(XL)
    
#(X, _), (_, _) = fashion_mnist.load_data()
# X = X[:40]
X_train = X / 127.5 - 1.
#X_train = np.expand_dims(X_train, axis=3)
image_helper = ImageHelper()
img_shape = X_train[0].shape
gan = DCGAN(img_shape, 10, image_helper, 1)
gan.train(20000, X_train, batch_size=32)


# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
