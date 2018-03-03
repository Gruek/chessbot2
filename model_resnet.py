from keras import models
from keras import layers
from keras import optimizers
import os.path
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import regularizers
from keras import initializers

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

# WEIGHTS_FILE = '/data/kru03a/chbot/data/model_resnet.h5'
WEIGHTS_FILE = 'data/model_resnet.h5'
WEIGHT_DECAY = 0.002

def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer='glorot_normal')(y)
    # y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer='glorot_normal')(y)
    # y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        # shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

def get_model(outputs):
    model = None
    if os.path.isfile(WEIGHTS_FILE) and False:
        model = models.load_model(WEIGHTS_FILE)
        print('model loaded')
    else:
        inp = layers.Input(shape=(8,8,12,))
        flat_inp = layers.Flatten()(inp)
        y = inp
        y = residual_block(y, 128, _project_shortcut=True)
        for i in range(25):
            y = residual_block(y, 128, _project_shortcut=False)
        y = layers.Flatten()(y)

        inp2 = layers.Input(shape=(4,))
        y = layers.concatenate([y, flat_inp, inp2])
        y = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y)
        y = layers.Dropout(0.1)(y)

        y_policy = layers.Dropout(0.2)(y)
        y_policy = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y_policy)
        y_policy = layers.Dropout(0.1)(y_policy)
        y_policy = layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y_policy)
        y_policy = layers.Dropout(0.1)(y_policy)
        y_policy = layers.Dense(outputs, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y_policy)
        y_policy = layers.Activation("softmax")(y_policy)

        y_value = layers.Dropout(0.2)(y)
        y_value = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y_value)
        y_value = layers.Dropout(0.1)(y_value)
        y_value = layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y_value)
        y_value = layers.Dropout(0.1)(y_value)
        y_value = layers.Dense(2, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y_value)
        y_value = layers.Activation("softmax")(y_value)

        model = models.Model(inputs=[inp, inp2], outputs=[y_policy, y_value])
        # model.summary()
        model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.6), loss='categorical_crossentropy', loss_weights=[1., 1.], metrics=['accuracy'])

        if os.path.isfile(WEIGHTS_FILE):
            model.load_weights(WEIGHTS_FILE)

    return model

def save_model(model):
    model.save(WEIGHTS_FILE)
