import numpy as np
import tensorflow as tf
from keras import layers
import keras
from keras import Input
from keras.models import load_model
import sklearn
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, concatenate
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras import regularizers
from keras.regularizers import l2


#The functions return our metric and loss
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef_loss(y_true, y_pred):
    return -1.0 * dice_coef(y_true, y_pred)


#3D U-Net
def Unet():
        dropout_rate = 0.2
    
        inputs = Input((64,64,64,1))
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(inputs)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(dropout_rate)(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool1)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(dropout_rate)(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool2)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Dropout(dropout_rate)(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool3)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Dropout(dropout_rate)(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool4)
        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Dropout(dropout_rate)(conv5)

        up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=4)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up6)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Dropout(dropout_rate)(conv6)

        up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=4)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Dropout(dropout_rate)(conv7)

        up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=4)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up8)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Dropout(dropout_rate)(conv8)

        up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=4)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up9)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Dropout(dropout_rate)(conv9)
    

        conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

        model = Model(inputs, conv10)


        return model

#3D ResU-Net
def ResUnet():
        dropout_rate = 0.2
    
        inputs = Input((64,64,64,1))
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(inputs)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(dropout_rate)(conv1)
        conv1 = concatenate([inputs, conv1], axis=4)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool1)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(dropout_rate)(conv2)
        conc2 = concatenate([pool1, conv2], axis=4)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool2)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Dropout(dropout_rate)(conv3)
        conc3 = concatenate([pool2, conv3], axis=4)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool3)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Dropout(dropout_rate)(conv4)
        conc4 = concatenate([pool3, conv4], axis=4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool4)
        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Dropout(dropout_rate)(conv5)
        conc5 = concatenate([pool4, conv5], axis=4)
        
        up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=4)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up6)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Dropout(dropout_rate)(conv6)
        conc6 = concatenate([up6, conv6], axis=4)

        up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=4)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Dropout(dropout_rate)(conv7)
        conc7 = concatenate([up7, conv7], axis=4)

        up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=4)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up8)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Dropout(dropout_rate)(conv8)
        conc8 = concatenate([up8, conv8], axis=4)

        up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=4)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up9)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Dropout(dropout_rate)(conv9)
        conc9 = concatenate([up9, conv9], axis=4)
    

        conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

        model = Model(inputs, conv10)


        return model

#3D DenseU-Net
def DenseUnet():
        dropout_rate = 0.2
    
        inputs = Input((64,64,64,1))
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(inputs)
        conv1 = concatenate([inputs, conv1], axis=4)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(dropout_rate)(conv1)
        conv1 = concatenate([inputs, conv1], axis=4)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool1)
        conc2 = concatenate([pool1, conv2], axis=4)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(dropout_rate)(conv2)
        conc2 = concatenate([pool1, conv2], axis=4)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool2)
        conc3 = concatenate([pool2, conv3], axis=4)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Dropout(dropout_rate)(conv3)
        conc3 = concatenate([pool2, conv3], axis=4)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool3)
        conc4 = concatenate([pool3, conv4], axis=4)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Dropout(dropout_rate)(conv4)
        conc4 = concatenate([pool3, conv4], axis=4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(pool4)
        conc5 = concatenate([pool4, conv5], axis=4)
        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Dropout(dropout_rate)(conv5)
        conc5 = concatenate([pool4, conv5], axis=4)
        
        up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=4)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up6)
        conc6 = concatenate([up6, conv6], axis=4)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Dropout(dropout_rate)(conv6)
        conc6 = concatenate([up6, conv6], axis=4)

        up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=4)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Dropout(dropout_rate)(conv7)
        conc7 = concatenate([up7, conv7], axis=4)

        up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=4)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up8)
        conc8 = concatenate([up8, conv8], axis=4)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Dropout(dropout_rate)(conv8)
        conc8 = concatenate([up8, conv8], axis=4)

        up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=4)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(up9)
        conc9 = concatenate([up9, conv9], axis=4)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same' ,kernel_regularizer=l2(1e-4))(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Dropout(dropout_rate)(conv9)
        conc9 = concatenate([up9, conv9], axis=4)
    

        conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

        model = Model(inputs, conv10)


        return model

#3D WideU-net and 3D U-netPP are written by referring to the code below.
#https://github.com/jiao133/Nested-UNet/blob/master/model.py

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    
    dropout_rate = 0.2
    bn_axis = 4

    act = 'relu'

    x = Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x

#3D WideU-Net
def WideUNet(color_type=1, num_class=1):

    nb_filter = [16,32,64,128,256]
    act = 'relu'

    inputs = Input((64,64,64,1))
    conv1_1 = standard_unit(inputs, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv3DTranspose(nb_filter[3], (2, 2, 2), strides=(2, 2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    model = Model(inputs, unet_output)

    return model

def UnetPP(color_type=1, num_class=1, deep_supervision=False):

    nb_filter = [16,32,64,128,256]
    act = 'relu'

    inputs = Input((64,64,64,1))
    conv1_1 = standard_unit(inputs, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool2')(conv2_1)

    up1_2 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool3')(conv3_1)

    up2_2 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool4')(conv4_1)

    up3_2 = Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv3DTranspose(nb_filter[0],(2, 2, 2), strides=(2, 2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv3DTranspose(nb_filter[3], (2, 2, 2), strides=(2, 2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(inputs, [nestnet_output_1,
                                nestnet_output_2,
                                nestnet_output_3,
                                nestnet_output_4])
    else:
        model = Model(inputs, [nestnet_output_4])

    return model