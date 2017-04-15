from keras.layers.core import Reshape, Lambda
from keras.layers import TimeDistributed
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, MaxPooling2D, MaxPooling3D, Input, merge, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.core import Dense, Dropout, Activation, Flatten, SpatialDropout3D
from keras.optimizers import SGD , Adam, RMSprop
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.noise import GaussianNoise

def create_model(name):
    if name=="model3" or name == "model3_mean" or name == "model3_nomean2":
        return create_model_3()
    if name=="model6":
        return create_model_6()
    if name=="model7":
        return create_model_6()
    if name=="model8":
        return create_model_8()
    if name=="model3_noise" or name == "model3_noise_comb" or name == "model3_noise_comb2" or name == "model3_noise_comb_dsb" or name == "model3_noise_comb3":
        return create_model_3_noise()
    if name == "model3_noise2":
        return create_model_3_noise2()

def create_model_3_noise2():
    inputs = Input((32, 32, 32, 1))

    noise = GaussianNoise(sigma=0.02)(inputs)
    
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(noise)
    conv1 = SpatialDropout3D(0.4)(conv1)
    conv1 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv1)

    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = SpatialDropout3D(0.4)(conv2)
    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2, 2))(conv2)

    x = Flatten()(pool2)
    x = Dense(128, init='normal')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, init='normal')(x)
    predictions = Dense(1, init='normal', activation='sigmoid')(x)
        
    model = Model(input=inputs, output=predictions)
    model.summary()
    optimizer = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy','precision','recall','mean_squared_error','accuracy'])

    return model
    


    
def create_model_3_noise():
    inputs = Input((32, 32, 32, 1))

    noise = GaussianNoise(sigma=0.05)(inputs)
    
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(noise)
    conv1 = SpatialDropout3D(0.1)(conv1)
    conv1 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv1)

    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = SpatialDropout3D(0.1)(conv2)
    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2, 2))(conv2)

    x = Flatten()(pool2)
    x = Dense(64, init='normal')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, init='normal', activation='sigmoid')(x)
        
    model = Model(input=inputs, output=predictions)
    model.summary()
    optimizer = Adam(lr=0.000001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy','precision','recall','mean_squared_error','accuracy'])

    return model
    

def create_model_8():
    inputs = Input((32, 32, 32, 1))

    #noise = GaussianNoise(sigma=0.1)(x)
    
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = SpatialDropout3D(0.2)(conv1)
    conv1 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    conv1 = SpatialDropout3D(0.2)(conv1)
    conv1 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv1)
    
    conv2 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = SpatialDropout3D(0.2)(conv2)
    conv2 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2, 2))(conv2)

    x = Flatten()(pool2)
    x = Dense(64, init='normal')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, init='normal', activation='sigmoid')(x)
        
    model = Model(input=inputs, output=predictions)
    model.summary()
    optimizer = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy','precision','recall','mean_squared_error','accuracy'])

    return model
    
    
def create_model_7():
    inputs = Input((32, 32, 32, 1))

    #noise = GaussianNoise(sigma=0.1)(x)
    
    conv1 = Convolution3D(32, 5, 5, 5, activation='relu', border_mode='same')(inputs)
    conv1 = SpatialDropout3D(0.1)(conv1)
    conv1 = Convolution3D(64, 5, 5, 5, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv1)

    conv2 = Convolution3D(128, 5, 5, 5, activation='relu', border_mode='same')(pool1)
    conv2 = SpatialDropout3D(0.1)(conv2)
    conv2 = Convolution3D(128, 5, 5, 5, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2, 2))(conv2)

    x = Flatten()(pool2)
    x = Dense(64, init='normal')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, init='normal', activation='sigmoid')(x)
        
    model = Model(input=inputs, output=predictions)
    model.summary()
    optimizer = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy','precision','recall','mean_squared_error','accuracy'])

    return model


def create_model_6():
    inputs = Input((32, 32, 32, 1))

    #noise = GaussianNoise(sigma=0.1)(x)
    
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = SpatialDropout3D(0.1)(conv1)

    conv1 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    conv1 = SpatialDropout3D(0.1)(conv1)

    conv1 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv1)

    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = SpatialDropout3D(0.1)(conv2)

    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    conv2 = SpatialDropout3D(0.1)(conv2)

    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2, 2))(conv2)

    x = Flatten()(pool2)
    x = Dense(64, init='normal')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, init='normal', activation='sigmoid')(x)
        
    model = Model(input=inputs, output=predictions)
    model.summary()
    optimizer = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy','precision','recall','mean_squared_error','accuracy'])

    return model


def create_model_5():
    inputs = Input((32, 32, 32, 1))

    noise = GaussianNoise(sigma=0.05)(inputs)
    
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(noise)
    conv1 = SpatialDropout3D(0.1)(conv1)
    conv1 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv1)

    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = SpatialDropout3D(0.1)(conv2)
    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2, 2))(conv2)

    x = Flatten()(pool2)
    x = Dense(64, init='normal')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, init='normal', activation='sigmoid')(x)
        
    model = Model(input=inputs, output=predictions)
    model.summary()
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy','precision','recall','mean_squared_error','accuracy'])

    return model
    


    
def create_model_4():
    inputs1 = Input((32, 32, 32, 1))
    inputs2 = Input((6,))

    #noise = GaussianNoise(sigma=0.1)(x)
    
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs1)
    conv1 = SpatialDropout3D(0.1)(conv1)
    conv1 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv1)

    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = SpatialDropout3D(0.1)(conv2)
    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2, 2))(conv2)

    x = Flatten()(pool2)
    x = merge([x, inputs2], mode='concat')
    x = Dense(64, init='normal')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, init='normal', activation='sigmoid')(x)
        
    model = Model(input=[inputs1,inputs2], output=predictions)
    model.summary()
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy','precision','recall','mean_squared_error','accuracy'])

    return model


def create_model_3():
    inputs = Input((32, 32, 32, 1))

    #noise = GaussianNoise(sigma=0.1)(x)
    
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = SpatialDropout3D(0.1)(conv1)
    conv1 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv1)

    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = SpatialDropout3D(0.1)(conv2)
    conv2 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2, 2))(conv2)

    x = Flatten()(pool2)
    x = Dense(64, init='normal')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, init='normal', activation='sigmoid')(x)
        
    model = Model(input=inputs, output=predictions)
    model.summary()
    optimizer = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy','precision','recall','mean_squared_error','accuracy'])

    return model



def create_model_2():
    inputs = Input((32, 32, 32, 1))

    #noise = GaussianNoise(sigma=0.1)(x)
    
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = SpatialDropout3D(0.1)(conv1)
    conv1 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv1)

    x = Flatten()(pool1)
    x = Dense(64, init='normal')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, init='normal', activation='sigmoid')(x)
        
    model = Model(input=inputs, output=predictions)
    model.summary()
    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy','precision','recall','mean_squared_error','accuracy'])

    return model


def create_model_1():
    inputs = Input((32, 32, 32, 1))

    #noise = GaussianNoise(sigma=0.1)(x)
    
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = SpatialDropout3D(0.1)(conv1)
    conv1 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv1)

    x = Flatten()(pool1)
    x = Dense(64, init='normal')(x)
    predictions = Dense(1, init='normal', activation='sigmoid')(x)
        
    model = Model(input=inputs, output=predictions)
    model.summary()
    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy','precision','recall','mean_squared_error','accuracy'])

    return model
