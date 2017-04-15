from keras.layers.core import Reshape, Lambda
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, MaxPooling2D, Input, merge, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.layers.convolutional import Convolution2D, SeparableConvolution2D, AtrousConvolution2D
from keras.layers.local import LocallyConnected2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.optimizers import SGD , Adam, RMSprop
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras import backend as K



# change the loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

'''
The UNET model is compiled in this function.
'''
def unet_model(trainable=True,embedding=False):
    print "creating model trainable=",trainable," embedding",embedding
    inputs = Input((512, 512, 1))
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', trainable=trainable)(inputs)
    #conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', trainable=trainable)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', trainable=trainable)(pool1)
    #conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', trainable=trainable)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', trainable=trainable)(pool2)
    #conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', trainable=trainable)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable=trainable)(pool3)
    #conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable=trainable)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', trainable=trainable)(pool4)
    #conv5 = Dropout(0.2)(conv5)

    if embedding: 
        conv5 = Convolution2D(64, 1, 1, name="embedding_64d",activation='relu', border_mode='same')(conv5)
        conv5 = Convolution2D(1024, 1, 1, name="embedding_1024u",activation='relu', border_mode='same')(conv5)

    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', trainable=trainable)(conv5)
    
    bridge4 = SpatialDropout2D(0.2)(conv4)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), bridge4], mode='concat')
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable=trainable)(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable=trainable)(conv6)

    bridge3 = SpatialDropout2D(0.2)(conv3)
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), bridge3], mode='concat')
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', trainable=trainable)(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', trainable=trainable)(conv7)

    bridge2 = SpatialDropout2D(0.2)(conv2)
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), bridge2], mode='concat')
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', trainable=trainable)(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', trainable=trainable)(conv8)

    bridge1 = SpatialDropout2D(0.2)(conv1)
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), bridge1], mode='concat')
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', trainable=trainable)(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', trainable=trainable)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
        
    predictions = Reshape(target_shape=(262144,1))(conv10)
        
    model = Model(input=inputs, output=predictions)
    model.summary()
    #optimizer = SGD(lr=0.00001, momentum=0.985, decay=0.0, nesterov=True)
    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[dice_coef,'binary_accuracy','precision','recall','mean_squared_error'],sample_weight_mode='temporal')

    return model

