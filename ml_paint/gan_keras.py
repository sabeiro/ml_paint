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
from gan_spine import gan_spine
 

class gan_deep(gan_spine):
    def __init__(self, opt):
        gan_spine.__init__(self, opt)

    def def_encoder_block(self,layer_in, n_filters, batchnorm=True):
        init = RandomNormal(stddev=0.02)
        g = Conv2D(n_filters,self.kernel,strides=(2,2),padding='same', kernel_initializer=init)(layer_in)
        if batchnorm: g = BatchNormalization()(g, training=True)
        g = LeakyReLU(alpha=0.2)(g)
        return g
 
    def decoder_block(self,layer_in,skip_in,n_filters,dropout=True):
        init = RandomNormal(stddev=0.02)
        g = Conv2DTranspose(n_filters,self.kernel,strides=(2,2),padding='same',kernel_initializer=init)(layer_in)
        g = BatchNormalization()(g,training=True)
        if dropout: g = Dropout(0.5)(g,training=True)
        g = Concatenate()([g, skip_in])
        g = Activation('relu')(g)
        return g
        
    def build_gen_model(self,optimizer):
        init = RandomNormal(stddev=0.02)
        in_image = Input(shape=self.img_shape)
        e1 = self.def_encoder_block(in_image, 64, batchnorm=False)
        e2 = self.def_encoder_block(e1, 128)
        e3 = self.def_encoder_block(e2, 256)
        e4 = self.def_encoder_block(e3, 512)
        e5 = self.def_encoder_block(e4, 512)
        # e6 = self.def_encoder_block(e5, 512)
        # e7 = self.def_encoder_block(e6, 512)
        b = Conv2D(512,self.kernel,strides=(2,2),padding='same',kernel_initializer=init)(e5)
        b = Activation('relu')(b)
        # d1 = self.decoder_block(b, e7, 512)
        # d2 = self.decoder_block(d1, e6, 512)
        d3 = self.decoder_block(b, e5, 512)
        d4 = self.decoder_block(d3, e4, 512, dropout=False)
        d5 = self.decoder_block(d4, e3, 256, dropout=False)
        d6 = self.decoder_block(d5, e2, 128, dropout=False)
        d7 = self.decoder_block(d6, e1, 64, dropout=False)
        g = Conv2DTranspose(3,self.kernel,strides=(2,2),padding='same',kernel_initializer=init)(d7)
        out_image = Activation('tanh')(g)
        self.gen_model = Model(in_image, out_image)
        plot_model(self.gen_model,to_file=self.baseDir+'model_gan/gen_model.png',show_shapes=True,show_layer_names=True)
        # self.gen_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        # print(self.gen_model.summary())
        
    def build_disc_model(self, optimizer):
        init = RandomNormal(stddev=0.02)
        in_src_image = Input(shape=self.img_shape)
        in_target_image = Input(shape=self.img_shape)
        merged = Concatenate()([in_src_image, in_target_image])
        d = Conv2D(64, self.kernel, strides=(2,2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, self.kernel, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(256, self.kernel, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(512, self.kernel, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(512, self.kernel, padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(1, self.kernel, padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
        self.disc_model = Model([in_src_image, in_target_image], patch_out)
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc_model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        #self.disc_model.trainable = False
        plot_model(self.disc_model,to_file=self.baseDir+'model_gan/disc_model.png',show_shapes=True,show_layer_names=True)
        # print(self.disc_model.summary())
        
class superRes(gan_spine):
    def __init__(self, opt):
        gan_spine.__init__(self, opt)

    def def_encoder_block(self,layer_in, n_filters, batchnorm=True):
        init = RandomNormal(stddev=0.02)
        g = Conv2D(n_filters,self.kernel,strides=(2,2),padding='same', kernel_initializer=init)(layer_in)
        if batchnorm: g = BatchNormalization()(g, training=True)
        g = LeakyReLU(alpha=0.2)(g)
        return g
 
    def decoder_block(self,layer_in,skip_in,n_filters,dropout=True):
        init = RandomNormal(stddev=0.02)
        g = Conv2DTranspose(n_filters,self.kernel,strides=(2,2),padding='same',kernel_initializer=init)(layer_in)
        g = BatchNormalization()(g,training=True)
        if dropout: g = Dropout(0.5)(g,training=True)
        g = Concatenate()([g, skip_in])
        g = Activation('relu')(g)
        return g
        
    def build_gen_model(self,optimizer):
        init = RandomNormal(stddev=0.02)
        in_image = Input(shape=self.img_shape)
        e1 = self.def_encoder_block(in_image, 64, batchnorm=False)
        e2 = self.def_encoder_block(e1, 128)
        e3 = self.def_encoder_block(e2, 256)
        e4 = self.def_encoder_block(e3, 512)
        e5 = self.def_encoder_block(e4, 512)
        # e6 = self.def_encoder_block(e5, 512)
        # e7 = self.def_encoder_block(e6, 512)
        b = Conv2D(512,self.kernel,strides=(2,2),padding='same',kernel_initializer=init)(e5)
        b = Activation('relu')(b)
        # d1 = self.decoder_block(b, e7, 512)
        # d2 = self.decoder_block(d1, e6, 512)
        d3 = self.decoder_block(b, e5, 512)
        d4 = self.decoder_block(d3, e4, 512, dropout=False)
        d5 = self.decoder_block(d4, e3, 256, dropout=False)
        d6 = self.decoder_block(d5, e2, 128, dropout=False)
        d7 = self.decoder_block(d6, e1, 64, dropout=False)
        d8 = self.decoder_block(d7, e1, 64, dropout=False)
        d9 = self.decoder_block(d8, e1, 64, dropout=False)
        g = Conv2DTranspose(3,self.kernel,strides=(2,2),padding='same',kernel_initializer=init)(d7)
        out_image = Activation('tanh')(g)
        self.gen_model = Model(in_image, out_image)
        plot_model(self.gen_model,to_file=self.baseDir+'model_gan/gen_model.png',show_shapes=True,show_layer_names=True)
        # self.gen_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        # print(self.gen_model.summary())
        
    def build_disc_model(self, optimizer):
        init = RandomNormal(stddev=0.02)
        in_src_image = Input(shape=self.img_shape)
        in_target_image = Input(shape=self.img_shape*4)
        merged = Concatenate()([in_src_image, in_target_image])
        d = Conv2D(64, self.kernel, strides=(2,2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, self.kernel, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(256, self.kernel, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(512, self.kernel, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(512, self.kernel, padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(1, self.kernel, padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
        self.disc_model = Model([in_src_image, in_target_image], patch_out)
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc_model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        #self.disc_model.trainable = False
        plot_model(self.disc_model,to_file=self.baseDir+'model_gan/disc_model.png',show_shapes=True,show_layer_names=True)
        # print(self.disc_model.summary())

        
class gan_keras(gan_spine):
    def __init__(self, opt):
        gan_spine.__init__(self, opt)
    
    def build_gen_model(self,optimizer):
        gen_input = Input(shape=self.img_shape)
        img_start = tuple([int(x/4.) for x in self.img_shape])
        gen_input = Input(shape=self.latent_dim)
        gen_seq = Dense(16*img_start[0]*img_start[1],activation="relu",input_dim=self.latent_dim)(gen_input)
        gen_seq = Reshape((img_start[0],img_start[1],16))(gen_seq)
        gen_seq = Dropout(0.20)(gen_seq)
        gen_seq = Conv2D(32,kernel_size=3,padding="same")(gen_seq)
        gen_seq = Conv2D(16,kernel_size=3,strides=2,input_shape=self.latent_dim,padding="same")(gen_seq)
        gen_seq = UpSampling2D()(gen_seq)
        # gen_seq = Conv2D(32, (1,1),kernel_initializer='he_normal', padding="same")(gen_input)
        gen_seq = BatchNormalization(momentum=0.8)(gen_seq)
        gen_seq = Activation("relu")(gen_seq)
        gen_seq = UpSampling2D()(gen_seq)
        gen_seq = Dropout(0.20)(gen_seq)
        gen_seq = Conv2D(32, kernel_size=3, padding="same")(gen_seq)
        gen_seq = BatchNormalization(momentum=0.8)(gen_seq)
        gen_seq = Activation("relu")(gen_seq)
        gen_seq = UpSampling2D()(gen_seq)
        # gen_seq = Conv2D(self.channels, kernel_size=3, padding="same")(gen_seq)
        gen_seq = Activation("tanh")(gen_seq)
        gen_model = Model(gen_input, gen_seq)
        self.gen_model = gen_model
        plot_model(gen_model,to_file=self.baseDir+'model_gan/gen_model.png',show_shapes=True,show_layer_names=True)
        # self.gen_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        # print(self.gen_model.summary())
        
    def build_disc_model(self, optimizer):
        disc_input = Input(shape=self.img_shape)
        disc_seq = Conv2D(16,kernel_size=3,strides=2,input_shape=self.img_shape,padding="same")(disc_input)
        disc_seq = LeakyReLU(alpha=0.2)(disc_seq)
        disc_seq = Dropout(0.25)(disc_seq)
        disc_seq = Conv2D(32, kernel_size=3, strides=2, padding="same")(disc_seq)
        disc_seq = ZeroPadding2D(padding=((0,1),(0,1)))(disc_seq)
        disc_seq = BatchNormalization(momentum=0.8)(disc_seq)
        disc_seq = LeakyReLU(alpha=0.2)(disc_seq)
        disc_seq = Dropout(0.25)(disc_seq)
        disc_seq = Conv2D(64, kernel_size=3, strides=2, padding="same")(disc_seq)
        disc_seq = BatchNormalization(momentum=0.8)(disc_seq)
        disc_seq = LeakyReLU(alpha=0.2)(disc_seq)
        disc_seq = Dropout(0.25)(disc_seq)
        disc_seq = Conv2D(128, kernel_size=3, strides=2, padding="same")(disc_seq)
        disc_seq = BatchNormalization(momentum=0.8)(disc_seq)
        disc_seq = LeakyReLU(alpha=0.2)(disc_seq)
        disc_seq = Dropout(0.25)(disc_seq)
        disc_seq = Flatten()(disc_seq)
        disc_seq = Dense(1, activation='sigmoid')(disc_seq)
        disc_model = Model(disc_input, disc_seq)
        self.disc_model = disc_model
        self.disc_model.compile(loss='mae',optimizer=optimizer,loss_weights=[0.5],metrics=['accuracy'])
        self.disc_model.trainable = False
        plot_model(disc_model,to_file=self.baseDir+'model_gan/disc_model.png',show_shapes=True,show_layer_names=True)
        # print(self.disc_model.summary())

class gan_keras2(gan_spine):
    def __init__(self, opt):
        gan_spine.__init__(self, opt)
    
    def build_gen_model(self,optimizer):
        gen_seq = Sequential(
                [Dense(128, input_dim=self.latent_dim),
                 BatchNormalization(momentum=0.8),
                 Dense(256),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(512),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(np.prod(self.img_shape), activation='tanh'),
                 Reshape(self.img_shape)])
        gen_output_tensor = gen_seq(gen_input)

        # keras.layers.Dense(7 * 7 * 128, input_shape =[num_features]), 
        # keras.layers.Reshape([7, 7, 128]), 
        # keras.layers.BatchNormalization(), 
        # keras.layers.Conv2DTranspose( 
        #     64, (5, 5), (2, 2), padding ="same", activation ="selu"), 
        # keras.layers.BatchNormalization(), 
        # keras.layers.Conv2DTranspose( 
        #     1, (5, 5), (2, 2), padding ="same", activation ="tanh"), 
        # model = Sequential()
        # n_nodes = 256 * 4 * 4
        # model.add(Dense(n_nodes, input_dim=self.latent_dim))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Reshape((4, 4, 256)))
        # model.add(Conv2DTranspose(128, self.kernel, strides=(2,2), padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(128, self.kernel, strides=(2,2), padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(128, self.kernel, strides=(2,2), padding='same'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
        # gen_model = model
        self.gen_model = gen_model
        plot_model(gen_model,to_file=self.baseDir+'model_gan/gen_model.png',show_shapes=True,show_layer_names=True)
        
    def build_disc_model(self, optimizer):
        disc_input = Input(shape=self.img_shape)
        model = Sequential()
        model.add(Conv2D(64, (3,3), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.trainable = False
        disc_model = model
        # keras.layers.Conv2D(64, (5, 5), (2, 2), padding ="same", input_shape =[28, 28, 1]), 
        # keras.layers.LeakyReLU(0.2), 
        # keras.layers.Dropout(0.3), 
        # keras.layers.Conv2D(128, (5, 5), (2, 2), padding ="same"), 
        # keras.layers.LeakyReLU(0.2), 
        # keras.layers.Dropout(0.3), 
        # keras.layers.Flatten(), 
        # keras.layers.Dense(1, activation ='sigmoid') 
        # disc_tensor = disc_seq(disc_input)
        self.disc_model = disc_model
        self.disc_model.compile(loss='binary_crossentropy',optimizer=optimizer,loss_weights=[0.5],metrics=['accuracy'])
        self.disc_model.trainable = False
        plot_model(disc_model,to_file=self.baseDir+'model_gan/disc_model.png',show_shapes=True,show_layer_names=True)
        # print(self.disc_model.summary())
        
        
class categorize():
    def __init__(self, opt):
        gan_spine.__init__(self, opt)
    
    def build_gen_model(self,optimizer):
        self.gen_model = None
        
    def build_disc_model(self, optimizer):
        class_num = 1
        disc_input = Input(shape=self.img_shape)
        disc_seq = Conv2D(32,kernel_size=3,strides=2,activation="relu"
                          ,input_shape=self.img_shape,padding="same")(disc_input)
        disc_seq = MaxPooling2D(2)(disc_seq)
        disc_seq = Dropout(0.2)(disc_seq)
        disc_seq = BatchNormalization()(disc_seq)
        disc_seq = Conv2D(128, 3, activation='relu', padding='same')(disc_seq)
        disc_seq = Dropout(0.2)(disc_seq)
        disc_seq = BatchNormalization()(disc_seq)
        disc_seq = Flatten()(disc_seq)
        disc_seq = Dropout(0.2)(disc_seq)
        disc_seq = Dense(32, activation='relu')(disc_seq)
        disc_seq = Dropout(0.3)(disc_seq)
        disc_seq = BatchNormalization()(disc_seq)
        disc_seq = Dense(class_num, activation='softmax')(disc_seq)
        self.disc_model = disc_model
        self.disc_model.compile(loss='categorical_crossentropy',optimizer=optimizer,loss_weights=[0.5],metrics=['accuracy', 'val_accuracy'])
        self.disc_model.trainable = False
        plot_model(disc_model,to_file=self.baseDir+'model_gan/disc_model.png',show_shapes=True,show_layer_names=True)

    def train_disc(self,X,y,epochs=25,batch_size=64):
        history = self.disc_model.fit(X_train,y_train,validation_data=(X_test,y_test)
                                      ,epochs=epochs,batch_size=batch_size)
        scores = self.disc_model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        pd.DataFrame(history.history).plot()
        plt.show()
