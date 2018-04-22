from keras import models
from keras import layers
from keras import optimizers
import os.path
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import regularizers
from keras import initializers
from keras import backend as K
from keras.utils import multi_gpu_model
import os

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
set_session(tf.Session(config=config))

WEIGHTS_FILE = 'data/model_densenet.h5'
TEMP_FILE = 'data/model_densenet.temp.h5'
# WEIGHTS_FILE = '/data/kru03a/chbot/data/model_densenet.h5'
# TEMP_FILE = '/data/kru03a/chbot/data/model_densenet.temp.h5'
WEIGHT_DECAY = 0.000

def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    # x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
    # x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    # x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def get_model(outputs=4184, num_gpus=1):
    # with tf.device('/device:GPU:0' if num_gpus == 1 else '/cpu:0'):
    if os.path.isfile(WEIGHTS_FILE) and False:
        model = models.load_model(WEIGHTS_FILE)
        print('model loaded')
    else:
        inp = layers.Input(shape=(8,8,12,))
        # flat_inp = layers.Flatten()(inp)
        y = inp

        y = layers.Conv2D(64, 1, use_bias=False, name='conv1/conv')(y)
        y = layers.Activation('relu', name='conv1/relu')(y)
        blocks = [6, 12, 48, 32] #densenet201
        for i, block in enumerate(blocks):
            y = dense_block(y, block, name='conv' + str(i + 2))
            y = transition_block(y, 0.5, name='pool' + str(i + 2))
        y = layers.GlobalAveragePooling2D(name='avg_pool')(y)
        # y = layers.Flatten()(y)

        inp2 = layers.Input(shape=(4,))
        y = layers.concatenate([y, inp2])
        y = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y)
        # y = layers.Dropout(0.1)(y)

        # y_policy = layers.Dropout(0.1)(y)
        y_policy = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y)
        # y_policy = layers.Dropout(0.1)(y_policy)
        y_policy = layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y_policy)
        # y_policy = layers.Dropout(0.1)(y_policy)
        y_policy = layers.Dense(outputs, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y_policy)
        y_policy = layers.Activation("softmax")(y_policy)

        # y_value = layers.Dropout(0.1)(y)
        y_value = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y)
        # y_value = layers.Dropout(0.1)(y_value)
        y_value = layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y_value)
        # y_value = layers.Dropout(0.1)(y_value)
        y_value = layers.Dense(2, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(y_value)
        y_value = layers.Activation("softmax")(y_value)

        model = models.Model(inputs=[inp, inp2], outputs=[y_policy, y_value])
        # model.summary()

        if os.path.isfile(WEIGHTS_FILE):
            model.load_weights(WEIGHTS_FILE)

    if num_gpus > 1:
        compiled_model = multi_gpu_model(model, gpus=num_gpus)
    else:
        compiled_model = model
    compiled_model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.1), loss='categorical_crossentropy', loss_weights=[1., 1.], metrics=['accuracy'])

    return compiled_model, model

def save_model(model):
    model.save(TEMP_FILE)
    os.replace(WEIGHTS_FILE, WEIGHTS_FILE + ".backup")
    os.replace(TEMP_FILE, WEIGHTS_FILE)
